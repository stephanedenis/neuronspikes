"""
Visualiseur 3D OpenGL pour les piles de neurones.

Affiche la hiérarchie de neurones en 3D avec:
- Chaque couche comme un plan horizontal
- Les neurones comme des sphères colorées
- Les connexions comme des lignes
- L'activité en luminosité/taille

Contrôles:
- Souris gauche + drag: Rotation
- Souris droite + drag: Zoom
- Souris milieu + drag: Pan
- R: Reset vue
- Space: Pause/Resume
- +/-: Vitesse de rotation auto
- Q/ESC: Quitter
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, List
import numpy as np

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

from .genesis import NeuronStack, NeuronLayer, Neuron, NeuronState


@dataclass
class Camera3D:
    """Caméra 3D pour la navigation."""
    # Position
    distance: float = 5.0
    rotation_x: float = 30.0  # Inclinaison
    rotation_y: float = 45.0  # Rotation horizontale
    pan_x: float = 0.0
    pan_y: float = 0.0
    
    # Limites
    min_distance: float = 1.0
    max_distance: float = 20.0
    
    # Animation
    auto_rotate: bool = True
    auto_rotate_speed: float = 10.0  # degrés/seconde
    
    def rotate(self, dx: float, dy: float):
        """Rotation par delta souris."""
        self.rotation_y += dx * 0.5
        self.rotation_x += dy * 0.5
        self.rotation_x = max(-89, min(89, self.rotation_x))
    
    def zoom(self, delta: float):
        """Zoom par delta."""
        self.distance *= (1.0 - delta * 0.1)
        self.distance = max(self.min_distance, min(self.max_distance, self.distance))
    
    def pan(self, dx: float, dy: float):
        """Pan par delta."""
        self.pan_x += dx * 0.01
        self.pan_y -= dy * 0.01
    
    def update(self, dt: float):
        """Mise à jour animation."""
        if self.auto_rotate:
            self.rotation_y += self.auto_rotate_speed * dt
    
    def apply(self):
        """Applique la transformation de caméra."""
        glTranslatef(0, 0, -self.distance)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        glTranslatef(self.pan_x, self.pan_y, 0)
    
    def reset(self):
        """Reset à la position par défaut."""
        self.distance = 5.0
        self.rotation_x = 30.0
        self.rotation_y = 45.0
        self.pan_x = 0.0
        self.pan_y = 0.0


@dataclass
class VisualizerConfig:
    """Configuration du visualiseur 3D."""
    # Fenêtre
    width: int = 1280
    height: int = 720
    title: str = "NeuronSpikes 3D Visualizer"
    
    # Espacement des couches
    layer_spacing: float = 0.8  # Distance verticale entre couches
    layer_scale: float = 2.0    # Échelle horizontale des couches
    
    # Neurones
    neuron_base_size: float = 0.02  # Taille de base
    neuron_active_scale: float = 2.0  # Multiplicateur quand actif
    
    # Couleurs (R, G, B, A)
    background_color: Tuple[float, ...] = (0.05, 0.05, 0.1, 1.0)
    grid_color: Tuple[float, ...] = (0.2, 0.2, 0.3, 0.5)
    connection_color: Tuple[float, ...] = (0.5, 0.5, 0.8, 0.3)
    
    # Performance
    max_neurons_per_layer: int = 500  # Limite pour le rendu
    update_interval: float = 1.0 / 60.0  # 60 fps


# Couleurs pour les états des neurones
NEURON_COLORS = {
    NeuronState.DORMANT: (0.3, 0.3, 0.3, 0.5),
    NeuronState.CHARGING: (0.8, 0.8, 0.2, 0.8),
    NeuronState.FIRING: (1.0, 0.2, 0.2, 1.0),
    NeuronState.REFRACTORY: (0.2, 0.2, 0.8, 0.6),
}

# Couleurs par couche
LAYER_COLORS = [
    (1.0, 0.4, 0.4),  # Rouge
    (0.4, 1.0, 0.4),  # Vert
    (0.4, 0.4, 1.0),  # Bleu
    (1.0, 1.0, 0.4),  # Jaune
    (1.0, 0.4, 1.0),  # Magenta
    (0.4, 1.0, 1.0),  # Cyan
    (1.0, 0.7, 0.4),  # Orange
    (0.7, 0.4, 1.0),  # Violet
]


class NeuronVisualizer3D:
    """Visualiseur 3D pour NeuronStack.
    
    Utilise OpenGL via GLUT pour le rendu temps réel.
    """
    
    def __init__(
        self,
        stack: Optional[NeuronStack] = None,
        config: Optional[VisualizerConfig] = None,
        update_callback: Optional[Callable[[], None]] = None
    ):
        """Initialise le visualiseur.
        
        Args:
            stack: Pile de neurones à visualiser
            config: Configuration du visualiseur
            update_callback: Callback appelé à chaque frame pour mettre à jour le stack
        """
        if not OPENGL_AVAILABLE:
            raise RuntimeError("PyOpenGL n'est pas installé. Installez avec: pip install PyOpenGL PyOpenGL-accelerate")
        
        self.stack = stack
        self.config = config or VisualizerConfig()
        self.update_callback = update_callback
        
        self.camera = Camera3D()
        self.paused = False
        self.last_time = time.time()
        
        # État souris
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_buttons = [False, False, False]
        
        # Statistiques
        self.frame_count = 0
        self.fps = 0.0
        self.fps_update_time = time.time()
        
        # Display lists pour optimisation
        self._sphere_list = None
        self._grid_list = None
        
        # Fenêtre initialisée?
        self._initialized = False
    
    def _init_gl(self):
        """Initialise OpenGL."""
        # Couleur de fond
        glClearColor(*self.config.background_color)
        
        # Depth test
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        
        # Blending pour transparence
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Éclairage
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1, 1, 1, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
        
        # Material
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Smooth shading
        glShadeModel(GL_SMOOTH)
        
        # Antialiasing
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        
        # Créer les display lists
        self._create_sphere_list()
        self._create_grid_list()
    
    def _create_sphere_list(self):
        """Crée une display list pour une sphère."""
        self._sphere_list = glGenLists(1)
        glNewList(self._sphere_list, GL_COMPILE)
        
        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluSphere(quadric, 1.0, 12, 8)
        gluDeleteQuadric(quadric)
        
        glEndList()
    
    def _create_grid_list(self):
        """Crée une display list pour la grille de base."""
        self._grid_list = glGenLists(1)
        glNewList(self._grid_list, GL_COMPILE)
        
        glBegin(GL_LINES)
        glColor4f(*self.config.grid_color)
        
        size = 2.0
        divisions = 10
        step = size * 2 / divisions
        
        for i in range(divisions + 1):
            pos = -size + i * step
            # Lignes X
            glVertex3f(-size, 0, pos)
            glVertex3f(size, 0, pos)
            # Lignes Z
            glVertex3f(pos, 0, -size)
            glVertex3f(pos, 0, size)
        
        glEnd()
        glEndList()
    
    def _reshape(self, width: int, height: int):
        """Callback de redimensionnement."""
        if height == 0:
            height = 1
        
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        aspect = width / height
        gluPerspective(45.0, aspect, 0.1, 100.0)
        
        glMatrixMode(GL_MODELVIEW)
    
    def _display(self):
        """Callback d'affichage."""
        # Temps
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # FPS
        self.frame_count += 1
        if current_time - self.fps_update_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.fps_update_time)
            self.frame_count = 0
            self.fps_update_time = current_time
        
        # Mise à jour
        if not self.paused:
            self.camera.update(dt)
            if self.update_callback:
                self.update_callback()
        
        # Clear
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Appliquer la caméra
        self.camera.apply()
        
        # Dessiner la grille de base
        glDisable(GL_LIGHTING)
        glCallList(self._grid_list)
        glEnable(GL_LIGHTING)
        
        # Dessiner les couches
        if self.stack:
            self._draw_stack()
        
        # HUD (texte 2D)
        self._draw_hud()
        
        glutSwapBuffers()
    
    def _draw_stack(self):
        """Dessine la pile de neurones."""
        num_layers = self.stack.num_layers
        
        for layer_idx, layer in enumerate(self.stack.layers):
            # Position Y de la couche
            y = layer_idx * self.config.layer_spacing
            
            # Couleur de la couche
            layer_color = LAYER_COLORS[layer_idx % len(LAYER_COLORS)]
            
            # Dessiner le plan de la couche (transparence)
            self._draw_layer_plane(layer, y, layer_color)
            
            # Dessiner les neurones
            self._draw_neurons(layer, y, layer_color)
        
        # Dessiner les connexions inter-couches
        self._draw_connections()
    
    def _draw_layer_plane(self, layer: NeuronLayer, y: float, color: Tuple[float, ...]):
        """Dessine le contour d'une couche (sans remplissage)."""
        h, w = layer.shape
        scale = self.config.layer_scale
        
        # Normaliser pour centrer
        half_w = (w / max(w, h)) * scale
        half_h = (h / max(w, h)) * scale
        
        # Contour fin seulement
        glDisable(GL_LIGHTING)
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        glColor4f(*color, 0.4)
        glVertex3f(-half_w, y, -half_h)
        glVertex3f(half_w, y, -half_h)
        glVertex3f(half_w, y, half_h)
        glVertex3f(-half_w, y, half_h)
        glEnd()
        glEnable(GL_LIGHTING)
    
    def _draw_neurons(self, layer: NeuronLayer, y: float, layer_color: Tuple[float, ...]):
        """Dessine les neurones d'une couche."""
        h, w = layer.shape
        scale = self.config.layer_scale
        
        neurons = layer.neurons[:self.config.max_neurons_per_layer]
        
        for neuron in neurons:
            # Position du centroïde
            cx, cy = neuron.centroid
            
            # Convertir en coordonnées 3D normalisées
            nx = (cx / w - 0.5) * 2 * scale * (w / max(w, h))
            nz = (cy / h - 0.5) * 2 * scale * (h / max(w, h))
            
            # Taille basée sur l'état
            size = self.config.neuron_base_size
            if neuron.state == NeuronState.FIRING:
                size *= self.config.neuron_active_scale
            elif neuron.state == NeuronState.CHARGING:
                size *= (1.0 + neuron.potential)
            
            # Couleur basée sur l'état
            state_color = NEURON_COLORS.get(neuron.state, (0.5, 0.5, 0.5, 0.5))
            
            # Mélanger avec la couleur de la couche
            final_color = (
                state_color[0] * 0.5 + layer_color[0] * 0.5,
                state_color[1] * 0.5 + layer_color[1] * 0.5,
                state_color[2] * 0.5 + layer_color[2] * 0.5,
                state_color[3]
            )
            
            # Dessiner la sphère
            glPushMatrix()
            glTranslatef(nx, y, nz)
            glScalef(size, size, size)
            glColor4f(*final_color)
            glCallList(self._sphere_list)
            glPopMatrix()
    
    def _draw_connections(self):
        """Dessine les connexions entre neurones de couches adjacentes.
        
        Les connexions montrent les relations hiérarchiques:
        - Un neurone de couche N+1 est connecté aux neurones de couche N
          dont les champs récepteurs se chevauchent avec le sien.
        """
        if self.stack.num_layers < 2:
            return
        
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(1.0)
        
        glBegin(GL_LINES)
        
        # Pour chaque paire de couches adjacentes
        for layer_idx in range(1, self.stack.num_layers):
            upper_layer = self.stack.layers[layer_idx]
            lower_layer = self.stack.layers[layer_idx - 1]
            
            y_upper = layer_idx * self.config.layer_spacing
            y_lower = (layer_idx - 1) * self.config.layer_spacing
            
            # Limiter le nombre de connexions pour la performance
            upper_neurons = upper_layer.neurons[:self.config.max_neurons_per_layer]
            lower_neurons = lower_layer.neurons[:self.config.max_neurons_per_layer]
            
            if not lower_neurons:
                continue
            
            for upper_neuron in upper_neurons:
                # Position du neurone supérieur
                ux, uy_centroid = upper_neuron.centroid
                h_up, w_up = upper_layer.shape
                nx_up = (ux / w_up - 0.5) * 2 * self.config.layer_scale * (w_up / max(w_up, h_up))
                nz_up = (uy_centroid / h_up - 0.5) * 2 * self.config.layer_scale * (h_up / max(w_up, h_up))
                
                # Trouver les neurones inférieurs qui se chevauchent
                for lower_neuron in lower_neurons:
                    # Calculer le chevauchement des RF
                    overlap = self._compute_rf_overlap(upper_neuron, lower_neuron, upper_layer.shape, lower_layer.shape)
                    
                    if overlap > 0.1:  # Seuil de chevauchement
                        # Position du neurone inférieur
                        lx, ly_centroid = lower_neuron.centroid
                        h_low, w_low = lower_layer.shape
                        nx_low = (lx / w_low - 0.5) * 2 * self.config.layer_scale * (w_low / max(w_low, h_low))
                        nz_low = (ly_centroid / h_low - 0.5) * 2 * self.config.layer_scale * (h_low / max(w_low, h_low))
                        
                        # Couleur basée sur la force de la connexion
                        alpha = min(0.8, overlap)
                        
                        # Gradient de couleur: vert (actif) ou gris (dormant)
                        if upper_neuron.state == NeuronState.FIRING or lower_neuron.state == NeuronState.FIRING:
                            glColor4f(0.2, 1.0, 0.3, alpha)  # Vert vif si actif
                        else:
                            glColor4f(*self.config.connection_color[:3], alpha * 0.5)
                        
                        # Dessiner la ligne
                        glVertex3f(nx_low, y_lower, nz_low)
                        glVertex3f(nx_up, y_upper, nz_up)
        
        glEnd()
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
    
    def _compute_rf_overlap(self, neuron_up: Neuron, neuron_low: Neuron, 
                            shape_up: tuple, shape_low: tuple) -> float:
        """Calcule le chevauchement spatial des champs récepteurs.
        
        Utilise une approximation basée sur la distance des centroïdes
        car les RF peuvent avoir des tailles différentes entre couches.
        """
        # Normaliser les positions des centroïdes
        cx_up, cy_up = neuron_up.centroid
        cx_low, cy_low = neuron_low.centroid
        
        # Normaliser à [0, 1]
        nx_up = cx_up / shape_up[1]
        ny_up = cy_up / shape_up[0]
        nx_low = cx_low / shape_low[1]
        ny_low = cy_low / shape_low[0]
        
        # Distance normalisée (max = sqrt(2) ≈ 1.414)
        dist = math.sqrt((nx_up - nx_low)**2 + (ny_up - ny_low)**2)
        
        # Convertir en chevauchement (proche = fort chevauchement)
        # Rayon effectif basé sur la taille du RF
        radius_up = math.sqrt(neuron_up._rf_size) / max(shape_up)
        radius_low = math.sqrt(neuron_low._rf_size) / max(shape_low)
        combined_radius = radius_up + radius_low
        
        if dist > combined_radius:
            return 0.0
        
        return 1.0 - (dist / combined_radius)
    
    def _draw_hud(self):
        """Dessine le HUD (texte 2D)."""
        # Passer en mode 2D
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.config.width, 0, self.config.height, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        # Texte
        y = self.config.height - 20
        
        self._draw_text(10, y, f"FPS: {self.fps:.1f}")
        y -= 15
        
        if self.stack:
            stats = self.stack.get_stats()
            self._draw_text(10, y, f"Layers: {stats['num_layers']}")
            y -= 15
            self._draw_text(10, y, f"Neurons: {stats['total_neurons']}")
            y -= 15
            self._draw_text(10, y, f"Frame: {stats['frame_count']}")
            y -= 15
            
            # Par couche
            patterns_per_layer = stats.get('patterns_per_layer', [])
            stable_per_layer = stats.get('stable_patterns_per_layer', [])
            
            for i, (count, spikes) in enumerate(zip(stats['neurons_per_layer'], stats['spikes_per_layer'])):
                color = LAYER_COLORS[i % len(LAYER_COLORS)]
                glColor3f(*color)
                
                # Afficher patterns si disponibles
                if i < len(patterns_per_layer):
                    patterns = patterns_per_layer[i]
                    stable = stable_per_layer[i] if i < len(stable_per_layer) else 0
                    self._draw_text(10, y, f"  L{i}: {count} neurons, {spikes} spikes | {patterns} patterns ({stable} stable)")
                else:
                    self._draw_text(10, y, f"  L{i}: {count} neurons, {spikes} spikes")
                y -= 12
        
        if self.paused:
            glColor3f(1, 0.5, 0)
            self._draw_text(self.config.width // 2 - 30, self.config.height - 20, "PAUSED")
        
        # Contrôles
        glColor3f(0.7, 0.7, 0.7)
        self._draw_text(10, 50, "Mouse: Rotate/Zoom/Pan")
        self._draw_text(10, 35, "Space: Pause  R: Reset  Q: Quit")
        self._draw_text(10, 20, "+/-: Auto-rotate speed")
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
    
    def _draw_text(self, x: float, y: float, text: str):
        """Dessine du texte à une position 2D."""
        glRasterPos2f(x, y)
        for char in text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))
    
    def _keyboard(self, key: bytes, x: int, y: int):
        """Callback clavier."""
        if key == b'q' or key == b'\x1b':  # Q ou ESC
            glutLeaveMainLoop()
        elif key == b' ':
            self.paused = not self.paused
        elif key == b'r':
            self.camera.reset()
        elif key == b'+' or key == b'=':
            self.camera.auto_rotate_speed += 5
        elif key == b'-':
            self.camera.auto_rotate_speed = max(0, self.camera.auto_rotate_speed - 5)
        elif key == b'a':
            self.camera.auto_rotate = not self.camera.auto_rotate
    
    def _mouse(self, button: int, state: int, x: int, y: int):
        """Callback souris (boutons)."""
        if button < 3:
            self.mouse_buttons[button] = (state == GLUT_DOWN)
        self.mouse_x = x
        self.mouse_y = y
        
        # Molette
        if button == 3:  # Scroll up
            self.camera.zoom(1)
        elif button == 4:  # Scroll down
            self.camera.zoom(-1)
    
    def _motion(self, x: int, y: int):
        """Callback souris (mouvement)."""
        dx = x - self.mouse_x
        dy = y - self.mouse_y
        self.mouse_x = x
        self.mouse_y = y
        
        if self.mouse_buttons[0]:  # Gauche: rotation
            self.camera.rotate(dx, dy)
            self.camera.auto_rotate = False
        elif self.mouse_buttons[2]:  # Droit: zoom
            self.camera.zoom(dy)
        elif self.mouse_buttons[1]:  # Milieu: pan
            self.camera.pan(dx, dy)
    
    def _idle(self):
        """Callback idle - demande un rafraîchissement."""
        glutPostRedisplay()
    
    def run(self):
        """Lance le visualiseur."""
        if self._initialized:
            return
        
        # Initialiser GLUT
        glutInit([])
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(self.config.width, self.config.height)
        glutCreateWindow(self.config.title.encode())
        
        # Initialiser OpenGL
        self._init_gl()
        
        # Callbacks
        glutDisplayFunc(self._display)
        glutReshapeFunc(self._reshape)
        glutKeyboardFunc(self._keyboard)
        glutMouseFunc(self._mouse)
        glutMotionFunc(self._motion)
        glutIdleFunc(self._idle)
        
        self._initialized = True
        
        print(f"\n=== {self.config.title} ===")
        print("Contrôles:")
        print("  Souris gauche: Rotation")
        print("  Souris droite: Zoom")
        print("  Souris milieu: Pan")
        print("  Molette: Zoom")
        print("  Space: Pause/Resume")
        print("  R: Reset vue")
        print("  A: Toggle auto-rotation")
        print("  +/-: Vitesse rotation")
        print("  Q/ESC: Quitter")
        print()
        
        # Lancer la boucle principale
        glutMainLoop()
    
    def set_stack(self, stack: NeuronStack):
        """Change la pile à visualiser."""
        self.stack = stack


def create_demo_stack() -> NeuronStack:
    """Crée une pile de démonstration avec des neurones."""
    from .temporal import TemporalPattern
    
    stack = NeuronStack(
        base_shape=(64, 64),
        num_layers=4,
        reduction_factor=0.5
    )
    
    # Ajouter des neurones de démo dans chaque couche
    import random
    random.seed(42)  # Déterministe
    
    for layer_idx, layer in enumerate(stack.layers):
        h, w = layer.shape
        num_neurons = max(5, 20 - layer_idx * 5)
        
        for i in range(num_neurons):
            # Position pseudo-aléatoire
            x = (i * 7 + layer_idx * 3) % (w - 5)
            y = (i * 11 + layer_idx * 5) % (h - 5)
            size = 3 + layer_idx
            
            signature = np.zeros((h, w), dtype=bool)
            y1, y2 = max(0, y), min(h, y + size)
            x1, x2 = max(0, x), min(w, x + size)
            signature[y1:y2, x1:x2] = True
            
            pattern = TemporalPattern(
                pattern_id=layer_idx * 100 + i,
                signature=signature,
                confidence=0.9,
                occurrences=20,
                first_seen=0,
                last_seen=10,
            )
            layer.create_neuron_from_pattern(pattern)
    
    return stack


def run_demo():
    """Lance une démo du visualiseur 3D."""
    print("Création de la pile de démonstration...")
    stack = create_demo_stack()
    
    stats = stack.get_stats()
    print(f"Stack créé: {stats['num_layers']} couches, {stats['total_neurons']} neurones")
    
    # Callback de mise à jour (simule l'activité)
    frame = [0]
    
    def update():
        frame[0] += 1
        # Simuler des entrées aléatoires pour voir l'activité
        if frame[0] % 5 == 0:
            h, w = stack.base_shape
            input_pattern = np.random.random((h, w)) > 0.95
            stack.process(input_pattern)
    
    # Créer et lancer le visualiseur
    viz = NeuronVisualizer3D(
        stack=stack,
        update_callback=update
    )
    
    viz.run()


if __name__ == "__main__":
    run_demo()
