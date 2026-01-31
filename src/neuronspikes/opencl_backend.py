"""
OpenCL Backend - Accélération GPU pour NeuronSpikes.

Utilise PyOpenCL pour accélérer les opérations critiques:
- Échantillonnage polaire de la fovéa
- Calcul de saillance (gradient + mouvement)
- Corrélation stéréo
- Détection de rotation (corrélation circulaire)

Préfère l'AMD RX 480 (8GB, 36 CUs) sur la GTX 750 Ti (2GB, 5 CUs).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, List
from dataclasses import dataclass
import math

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    cl = None
    cl_array = None


@dataclass
class DeviceInfo:
    """Informations sur un périphérique OpenCL."""
    name: str
    platform: str
    device_type: str
    compute_units: int
    global_mem_mb: int
    local_mem_kb: int
    max_work_group_size: int


# Kernels OpenCL
KERNELS_SOURCE = """
// =============================================================================
// Kernel: Échantillonnage polaire de la fovéa
// =============================================================================
__kernel void polar_sample(
    __global const uchar* image,      // Image source (grayscale H×W)
    const int img_width,
    const int img_height,
    const float gaze_x,               // Point de fixation X
    const float gaze_y,               // Point de fixation Y
    const float rotation,             // Rotation theta (radians)
    __global const float* cell_params, // [inner_r, outer_r, start_angle, end_angle] × n_cells
    const int num_cells,
    __global float* activations       // Sortie: activation par cellule
) {
    int cell_id = get_global_id(0);
    if (cell_id >= num_cells) return;
    
    // Paramètres de la cellule
    int param_offset = cell_id * 4;
    float inner_r = cell_params[param_offset + 0];
    float outer_r = cell_params[param_offset + 1];
    float start_angle = cell_params[param_offset + 2] + rotation;
    float end_angle = cell_params[param_offset + 3] + rotation;
    
    // Échantillonner au centre de la cellule
    float r = (inner_r + outer_r) * 0.5f;
    float angle = (start_angle + end_angle) * 0.5f;
    
    // Convertir en coordonnées cartésiennes
    float x = gaze_x + r * cos(angle);
    float y = gaze_y + r * sin(angle);
    
    // Vérifier les bornes
    int ix = (int)x;
    int iy = (int)y;
    
    if (ix >= 0 && ix < img_width && iy >= 0 && iy < img_height) {
        activations[cell_id] = (float)image[iy * img_width + ix] / 255.0f;
    } else {
        activations[cell_id] = 0.0f;
    }
}

// =============================================================================
// Kernel: Échantillonnage polaire multi-points (meilleure qualité)
// =============================================================================
__kernel void polar_sample_multipoint(
    __global const uchar* image,
    const int img_width,
    const int img_height,
    const float gaze_x,
    const float gaze_y,
    const float rotation,
    __global const float* cell_params,
    const int num_cells,
    const int samples_per_cell,        // Nombre de points par cellule
    __global float* activations
) {
    int cell_id = get_global_id(0);
    if (cell_id >= num_cells) return;
    
    int param_offset = cell_id * 4;
    float inner_r = cell_params[param_offset + 0];
    float outer_r = cell_params[param_offset + 1];
    float start_angle = cell_params[param_offset + 2] + rotation;
    float end_angle = cell_params[param_offset + 3] + rotation;
    
    float total = 0.0f;
    int valid_samples = 0;
    
    // Échantillonner plusieurs points dans la cellule
    for (int i = 0; i < samples_per_cell; i++) {
        float t = (float)(i + 0.5f) / (float)samples_per_cell;
        float r = inner_r + t * (outer_r - inner_r);
        
        for (int j = 0; j < 3; j++) {
            float ta = (float)(j + 0.5f) / 3.0f;
            float angle = start_angle + ta * (end_angle - start_angle);
            
            float x = gaze_x + r * cos(angle);
            float y = gaze_y + r * sin(angle);
            
            int ix = (int)x;
            int iy = (int)y;
            
            if (ix >= 0 && ix < img_width && iy >= 0 && iy < img_height) {
                total += (float)image[iy * img_width + ix];
                valid_samples++;
            }
        }
    }
    
    if (valid_samples > 0) {
        activations[cell_id] = total / (valid_samples * 255.0f);
    } else {
        activations[cell_id] = 0.0f;
    }
}

// =============================================================================
// Kernel: Calcul de saillance (gradient Sobel)
// =============================================================================
__kernel void compute_saliency(
    __global const uchar* image,
    const int width,
    const int height,
    __global float* saliency
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        if (x < width && y < height) {
            saliency[y * width + x] = 0.0f;
        }
        return;
    }
    
    // Sobel X
    float gx = -1.0f * image[(y-1) * width + (x-1)]
             + -2.0f * image[y * width + (x-1)]
             + -1.0f * image[(y+1) * width + (x-1)]
             +  1.0f * image[(y-1) * width + (x+1)]
             +  2.0f * image[y * width + (x+1)]
             +  1.0f * image[(y+1) * width + (x+1)];
    
    // Sobel Y
    float gy = -1.0f * image[(y-1) * width + (x-1)]
             + -2.0f * image[(y-1) * width + x]
             + -1.0f * image[(y-1) * width + (x+1)]
             +  1.0f * image[(y+1) * width + (x-1)]
             +  2.0f * image[(y+1) * width + x]
             +  1.0f * image[(y+1) * width + (x+1)];
    
    // Magnitude du gradient normalisée
    saliency[y * width + x] = sqrt(gx * gx + gy * gy) / (255.0f * 4.0f);
}

// =============================================================================
// Kernel: Différence absolue (détection de mouvement)
// =============================================================================
__kernel void abs_diff(
    __global const uchar* img1,
    __global const uchar* img2,
    const int size,
    __global float* diff
) {
    int i = get_global_id(0);
    if (i >= size) return;
    
    diff[i] = fabs((float)img1[i] - (float)img2[i]) / 255.0f;
}

// =============================================================================
// Kernel: Corrélation stéréo entre deux ensembles d'activations
// =============================================================================
__kernel void stereo_correlation(
    __global const float* left_act,
    __global const float* right_act,
    const int num_cells,
    __global float* correlation_out,  // Produit élément par élément
    __global float* disparity_out     // Différence
) {
    int i = get_global_id(0);
    if (i >= num_cells) return;
    
    float l = left_act[i];
    float r = right_act[i];
    
    correlation_out[i] = l * r;
    disparity_out[i] = l - r;
}

// =============================================================================
// Kernel: Détection de rotation (corrélation circulaire sur les anneaux)
// =============================================================================
__kernel void detect_rotation_shift(
    __global const float* current_act,   // Activations courantes [rings × sectors]
    __global const float* prev_act,      // Activations précédentes
    const int num_rings,
    const int num_sectors,
    const int max_shift,                  // Décalage max à tester (±max_shift)
    __global float* correlations          // Sortie: corrélation pour chaque shift
) {
    int shift = get_global_id(0) - max_shift;  // De -max_shift à +max_shift
    if (shift < -max_shift || shift > max_shift) return;
    
    float correlation = 0.0f;
    
    for (int ring = 0; ring < num_rings; ring++) {
        for (int sector = 0; sector < num_sectors; sector++) {
            // Secteur décalé (avec wrap-around)
            int shifted_sector = (sector + shift + num_sectors) % num_sectors;
            
            int idx_current = ring * num_sectors + sector;
            int idx_shifted = ring * num_sectors + shifted_sector;
            
            correlation += current_act[idx_current] * prev_act[idx_shifted];
        }
    }
    
    correlations[shift + max_shift] = correlation;
}

// =============================================================================
// Kernel: Réduction pour trouver les pics de saillance
// =============================================================================
__kernel void find_saliency_peaks(
    __global const float* saliency,
    const int width,
    const int height,
    const int block_size,              // Taille des blocs pour réduction
    const float threshold,
    __global float* peak_values,       // Valeur max par bloc
    __global int* peak_positions       // Position (x, y) par bloc
) {
    int bx = get_global_id(0);
    int by = get_global_id(1);
    
    int start_x = bx * block_size;
    int start_y = by * block_size;
    int end_x = min(start_x + block_size, width);
    int end_y = min(start_y + block_size, height);
    
    float max_val = 0.0f;
    int max_x = start_x;
    int max_y = start_y;
    
    for (int y = start_y; y < end_y; y++) {
        for (int x = start_x; x < end_x; x++) {
            float val = saliency[y * width + x];
            if (val > max_val) {
                max_val = val;
                max_x = x;
                max_y = y;
            }
        }
    }
    
    int num_blocks_x = (width + block_size - 1) / block_size;
    int block_id = by * num_blocks_x + bx;
    
    peak_values[block_id] = (max_val > threshold) ? max_val : 0.0f;
    peak_positions[block_id * 2] = max_x;
    peak_positions[block_id * 2 + 1] = max_y;
}
"""


class OpenCLBackend:
    """Backend OpenCL pour accélération GPU.
    
    Utilise de préférence l'AMD RX 480 (plus puissante).
    """
    
    def __init__(self, prefer_amd: bool = True, verbose: bool = True):
        """Initialise le backend OpenCL.
        
        Args:
            prefer_amd: Préférer les GPU AMD (plus de mémoire/CUs disponibles)
            verbose: Afficher les informations de configuration
        """
        if not OPENCL_AVAILABLE:
            raise RuntimeError("PyOpenCL n'est pas installé")
        
        self.prefer_amd = prefer_amd
        self.verbose = verbose
        
        # Sélectionner le meilleur périphérique
        self.device, self.platform = self._select_device()
        self.device_info = self._get_device_info()
        
        if verbose:
            print(f"OpenCL: {self.device_info.name}")
            print(f"  Platform: {self.device_info.platform}")
            print(f"  Compute Units: {self.device_info.compute_units}")
            print(f"  Memory: {self.device_info.global_mem_mb} MB")
        
        # Créer le contexte et la queue
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)
        
        # Compiler les kernels
        self.program = cl.Program(self.ctx, KERNELS_SOURCE).build()
        
        # Pré-récupérer les kernels pour éviter les warnings
        self.kernel_polar_sample = cl.Kernel(self.program, 'polar_sample')
        self.kernel_polar_sample_multipoint = cl.Kernel(self.program, 'polar_sample_multipoint')
        self.kernel_compute_saliency = cl.Kernel(self.program, 'compute_saliency')
        self.kernel_abs_diff = cl.Kernel(self.program, 'abs_diff')
        self.kernel_stereo_correlation = cl.Kernel(self.program, 'stereo_correlation')
        self.kernel_detect_rotation = cl.Kernel(self.program, 'detect_rotation_shift')
        
        # Cache des buffers
        self._buffer_cache = {}
    
    def _select_device(self) -> Tuple:
        """Sélectionne le meilleur périphérique GPU.
        
        Returns:
            Tuple (device, platform)
        """
        best_device = None
        best_platform = None
        best_score = -1
        
        for platform in cl.get_platforms():
            for device in platform.get_devices():
                # Ne considérer que les GPU
                if device.type != cl.device_type.GPU:
                    continue
                
                # Score basé sur compute units et mémoire
                score = device.max_compute_units * 100
                score += device.global_mem_size // (1024 * 1024 * 100)
                
                # Bonus pour AMD si préféré
                if self.prefer_amd and "AMD" in device.name:
                    score += 1000
                
                # Bonus pour rusticl (meilleur support AMD)
                if "rusticl" in platform.name.lower():
                    score += 500
                
                if score > best_score:
                    best_score = score
                    best_device = device
                    best_platform = platform
        
        if best_device is None:
            # Fallback sur n'importe quel périphérique
            platform = cl.get_platforms()[0]
            best_device = platform.get_devices()[0]
            best_platform = platform
        
        return best_device, best_platform
    
    def _get_device_info(self) -> DeviceInfo:
        """Récupère les informations du périphérique."""
        d = self.device
        
        dtype = "GPU" if d.type == cl.device_type.GPU else "CPU"
        
        return DeviceInfo(
            name=d.name,
            platform=self.platform.name,
            device_type=dtype,
            compute_units=d.max_compute_units,
            global_mem_mb=d.global_mem_size // (1024 * 1024),
            local_mem_kb=d.local_mem_size // 1024,
            max_work_group_size=d.max_work_group_size,
        )
    
    def _get_buffer(self, name: str, size: int, flags=cl.mem_flags.READ_WRITE) -> cl.Buffer:
        """Récupère ou crée un buffer du cache."""
        key = (name, size)
        if key not in self._buffer_cache:
            self._buffer_cache[key] = cl.Buffer(self.ctx, flags, size)
        return self._buffer_cache[key]
    
    def polar_sample(
        self,
        image: NDArray[np.uint8],
        gaze_x: float,
        gaze_y: float,
        rotation: float,
        cell_params: NDArray[np.float32],
        multipoint: bool = False,
        samples_per_cell: int = 4
    ) -> NDArray[np.float32]:
        """Échantillonne l'image selon la géométrie polaire.
        
        Args:
            image: Image grayscale (H×W)
            gaze_x, gaze_y: Point de fixation
            rotation: Rotation theta
            cell_params: Paramètres des cellules [inner_r, outer_r, start_angle, end_angle] × N
            multipoint: Utiliser l'échantillonnage multi-points
            samples_per_cell: Nombre de samples par cellule (si multipoint)
            
        Returns:
            Activations des cellules
        """
        h, w = image.shape
        num_cells = len(cell_params) // 4
        
        # Créer les buffers
        img_buf = cl.Buffer(
            self.ctx, 
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=np.ascontiguousarray(image)
        )
        params_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=cell_params.astype(np.float32)
        )
        activations = np.zeros(num_cells, dtype=np.float32)
        act_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.WRITE_ONLY,
            activations.nbytes
        )
        
        # Exécuter le kernel
        if multipoint:
            self.kernel_polar_sample_multipoint.set_args(
                img_buf, np.int32(w), np.int32(h),
                np.float32(gaze_x), np.float32(gaze_y), np.float32(rotation),
                params_buf, np.int32(num_cells), np.int32(samples_per_cell),
                act_buf
            )
            cl.enqueue_nd_range_kernel(self.queue, self.kernel_polar_sample_multipoint, (num_cells,), None)
        else:
            self.kernel_polar_sample.set_args(
                img_buf, np.int32(w), np.int32(h),
                np.float32(gaze_x), np.float32(gaze_y), np.float32(rotation),
                params_buf, np.int32(num_cells),
                act_buf
            )
            cl.enqueue_nd_range_kernel(self.queue, self.kernel_polar_sample, (num_cells,), None)
        
        # Récupérer les résultats
        cl.enqueue_copy(self.queue, activations, act_buf)
        
        return activations
    
    def compute_saliency(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Calcule la carte de saillance GPU.
        
        Args:
            image: Image grayscale
            
        Returns:
            Carte de saillance normalisée
        """
        h, w = image.shape
        
        img_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=np.ascontiguousarray(image)
        )
        saliency = np.zeros((h, w), dtype=np.float32)
        sal_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, saliency.nbytes)
        
        # Exécuter le kernel 2D
        global_size = (w, h)
        self.kernel_compute_saliency.set_args(
            img_buf, np.int32(w), np.int32(h),
            sal_buf
        )
        cl.enqueue_nd_range_kernel(self.queue, self.kernel_compute_saliency, global_size, None)
        
        cl.enqueue_copy(self.queue, saliency, sal_buf)
        
        return saliency
    
    def abs_diff(
        self, 
        img1: NDArray[np.uint8], 
        img2: NDArray[np.uint8]
    ) -> NDArray[np.float32]:
        """Calcule la différence absolue (détection de mouvement).
        
        Args:
            img1, img2: Images à comparer
            
        Returns:
            Différence normalisée
        """
        size = img1.size
        
        buf1 = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=np.ascontiguousarray(img1.ravel())
        )
        buf2 = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=np.ascontiguousarray(img2.ravel())
        )
        diff = np.zeros(size, dtype=np.float32)
        diff_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, diff.nbytes)
        
        self.kernel_abs_diff.set_args(
            buf1, buf2, np.int32(size), diff_buf
        )
        cl.enqueue_nd_range_kernel(self.queue, self.kernel_abs_diff, (size,), None)
        
        cl.enqueue_copy(self.queue, diff, diff_buf)
        
        return diff.reshape(img1.shape)
    
    def stereo_correlation(
        self,
        left_act: NDArray[np.float32],
        right_act: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Calcule la corrélation et disparité stéréo.
        
        Args:
            left_act: Activations fovéa gauche
            right_act: Activations fovéa droite
            
        Returns:
            Tuple (corrélation, disparité)
        """
        num_cells = left_act.size
        
        left_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=left_act.astype(np.float32).ravel()
        )
        right_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=right_act.astype(np.float32).ravel()
        )
        
        correlation = np.zeros(num_cells, dtype=np.float32)
        disparity = np.zeros(num_cells, dtype=np.float32)
        corr_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, correlation.nbytes)
        disp_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, disparity.nbytes)
        
        self.kernel_stereo_correlation.set_args(
            left_buf, right_buf, np.int32(num_cells),
            corr_buf, disp_buf
        )
        cl.enqueue_nd_range_kernel(self.queue, self.kernel_stereo_correlation, (num_cells,), None)
        
        cl.enqueue_copy(self.queue, correlation, corr_buf)
        cl.enqueue_copy(self.queue, disparity, disp_buf)
        
        return correlation.reshape(left_act.shape), disparity.reshape(left_act.shape)
    
    def detect_rotation(
        self,
        current_act: NDArray[np.float32],
        prev_act: NDArray[np.float32],
        num_rings: int,
        num_sectors: int,
        max_shift: int = 4
    ) -> int:
        """Détecte le décalage de rotation optimal.
        
        Args:
            current_act: Activations courantes
            prev_act: Activations précédentes
            num_rings, num_sectors: Dimensions de la grille
            max_shift: Décalage maximum à tester
            
        Returns:
            Décalage optimal (en secteurs)
        """
        num_shifts = 2 * max_shift + 1
        
        curr_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=current_act.astype(np.float32).ravel()
        )
        prev_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=prev_act.astype(np.float32).ravel()
        )
        
        correlations = np.zeros(num_shifts, dtype=np.float32)
        corr_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, correlations.nbytes)
        
        self.kernel_detect_rotation.set_args(
            curr_buf, prev_buf,
            np.int32(num_rings), np.int32(num_sectors), np.int32(max_shift),
            corr_buf
        )
        cl.enqueue_nd_range_kernel(self.queue, self.kernel_detect_rotation, (num_shifts,), None)
        
        cl.enqueue_copy(self.queue, correlations, corr_buf)
        
        # Trouver le meilleur décalage
        best_idx = np.argmax(correlations)
        return best_idx - max_shift
    
    def get_stats(self) -> dict:
        """Retourne les statistiques du backend."""
        return {
            'available': True,
            'device': self.device_info.name,
            'platform': self.device_info.platform,
            'compute_units': self.device_info.compute_units,
            'memory_mb': self.device_info.global_mem_mb,
            'buffers_cached': len(self._buffer_cache),
        }


# Singleton global pour le backend
_backend: Optional[OpenCLBackend] = None


def get_opencl_backend(prefer_amd: bool = True, verbose: bool = True) -> Optional[OpenCLBackend]:
    """Récupère ou crée le backend OpenCL singleton.
    
    Args:
        prefer_amd: Préférer les GPU AMD
        verbose: Afficher les informations
        
    Returns:
        Backend OpenCL ou None si non disponible
    """
    global _backend
    
    if _backend is None:
        if not OPENCL_AVAILABLE:
            if verbose:
                print("OpenCL: Non disponible (pyopencl non installé)")
            return None
        
        try:
            _backend = OpenCLBackend(prefer_amd=prefer_amd, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"OpenCL: Erreur d'initialisation: {e}")
            return None
    
    return _backend


def is_opencl_available() -> bool:
    """Vérifie si OpenCL est disponible."""
    return OPENCL_AVAILABLE


def list_opencl_devices() -> List[DeviceInfo]:
    """Liste tous les périphériques OpenCL disponibles."""
    if not OPENCL_AVAILABLE:
        return []
    
    devices = []
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            dtype = "GPU" if device.type == cl.device_type.GPU else "CPU"
            devices.append(DeviceInfo(
                name=device.name,
                platform=platform.name,
                device_type=dtype,
                compute_units=device.max_compute_units,
                global_mem_mb=device.global_mem_size // (1024 * 1024),
                local_mem_kb=device.local_mem_size // 1024,
                max_work_group_size=device.max_work_group_size,
            ))
    
    return devices
