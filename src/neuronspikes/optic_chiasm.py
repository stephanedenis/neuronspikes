"""
Chiasma Optique - Croisement des fibres visuelles.

Implémente le croisement partiel bio-fidèle des voies visuelles:
- Fibres nasales: croisent vers l'hémisphère controlatéral
- Fibres temporales: restent dans l'hémisphère ipsilatéral

Résultat: Chaque hémisphère reçoit l'information d'un CHAMP VISUEL complet
(pas d'un œil complet), ce qui est essentiel pour:
- Fusion binoculaire
- Perception de la profondeur
- Correspondance stéréo

Anatomie:
    Œil gauche                      Œil droit
      Nasal → croise                  Nasal → croise
      Temporal → ipsi                 Temporal → ipsi
      
              ↓ Chiasma optique ↓
              
    Hémisphère DROIT:               Hémisphère GAUCHE:
      - Temporal gauche (ipsi)        - Temporal droit (ipsi)  
      - Nasal droit (contra)          - Nasal gauche (contra)
      = Champ visuel GAUCHE           = Champ visuel DROIT

En coordonnées polaires (fovéa):
    - Secteurs 0 à N/2-1: hémichamp temporal (côté externe)
    - Secteurs N/2 à N-1: hémichamp nasal (côté interne/nez)
    
    Pour l'œil GAUCHE: temporal = droite du champ = secteurs 0-15
    Pour l'œil DROIT:  temporal = gauche du champ = secteurs 16-31
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict
import numpy as np
from numpy.typing import NDArray


@dataclass
class ChiasmConfig:
    """Configuration du chiasma optique."""
    
    # Nombre de secteurs dans la fovéa
    num_sectors: int = 32
    
    # Nombre d'anneaux
    num_rings: int = 48
    
    # Pourcentage de fibres qui croisent (normalement ~53% nasales)
    # En pratique, on simplifie à 50% (hémichamp)
    crossing_ratio: float = 0.5
    
    # Activer le mélange progressif à la frontière nasal/temporal
    # (la fovéa centrale projette bilatéralement dans la réalité)
    foveal_bilateral: bool = True
    
    # Nombre d'anneaux fovéaux avec projection bilatérale
    foveal_bilateral_rings: int = 4


@dataclass
class HemifieldData:
    """Données d'un hémichamp visuel après le chiasma.
    
    Chaque hémisphère reçoit un hémichamp complet composé de:
    - Partie temporale de l'œil ipsilatéral
    - Partie nasale de l'œil controlatéral
    """
    # Données fusionnées de l'hémichamp (rings × sectors/2)
    data: NDArray[np.float32]
    
    # Composante temporale (œil ipsilatéral)
    temporal: NDArray[np.float32]
    
    # Composante nasale (œil controlatéral)  
    nasal: NDArray[np.float32]
    
    # Disparité calculée entre temporal et nasal
    disparity: NDArray[np.float32]
    
    # Corrélation entre temporal et nasal (qualité de la fusion)
    correlation: NDArray[np.float32]


class OpticChiasm:
    """Chiasma optique - réorganise les données rétiniennes par hémichamp.
    
    Transforme les données (œil gauche, œil droit) en (hémichamp gauche, hémichamp droit)
    pour permettre une vraie fusion binoculaire par correspondance spatiale.
    
    Exemple d'utilisation:
        chiasm = OpticChiasm(ChiasmConfig(num_sectors=32, num_rings=48))
        
        # Données des deux yeux (rings × sectors)
        left_eye = left_fovea.activations
        right_eye = right_fovea.activations
        
        # Réorganisation par hémichamp
        left_hemifield, right_hemifield = chiasm.process(left_eye, right_eye)
        
        # Maintenant left_hemifield contient le champ visuel DROIT
        # vu par les deux yeux (temporal droit + nasal gauche)
    """
    
    def __init__(self, config: ChiasmConfig = None):
        """Initialise le chiasma optique.
        
        Args:
            config: Configuration du chiasma
        """
        self.config = config or ChiasmConfig()
        
        # Indices des secteurs temporaux et nasaux pour chaque œil
        half = self.config.num_sectors // 2
        
        # Pour l'œil GAUCHE:
        # - Temporal (externe) = côté droit du champ = secteurs 0 à half-1
        # - Nasal (interne) = côté gauche du champ = secteurs half à N-1
        self.left_temporal_sectors = np.arange(0, half)
        self.left_nasal_sectors = np.arange(half, self.config.num_sectors)
        
        # Pour l'œil DROIT:
        # - Temporal (externe) = côté gauche du champ = secteurs half à N-1
        # - Nasal (interne) = côté droit du champ = secteurs 0 à half-1
        self.right_temporal_sectors = np.arange(half, self.config.num_sectors)
        self.right_nasal_sectors = np.arange(0, half)
        
        # Poids pour la zone fovéale bilatérale
        if self.config.foveal_bilateral:
            self._compute_bilateral_weights()
        
        # Statistiques
        self._frame_count = 0
        self._total_disparity = 0.0
        self._total_correlation = 0.0
    
    def _compute_bilateral_weights(self):
        """Calcule les poids pour la projection bilatérale fovéale.
        
        Dans la réalité, les ~2° centraux de la fovéa projettent
        vers les DEUX hémisphères (pas de division nette).
        """
        n_rings = self.config.num_rings
        n_bilateral = self.config.foveal_bilateral_rings
        
        # Poids décroissant du centre vers la périphérie
        self.bilateral_weights = np.zeros(n_rings, dtype=np.float32)
        for i in range(n_bilateral):
            # Décroissance linéaire
            self.bilateral_weights[i] = 1.0 - (i / n_bilateral)
    
    def process(
        self,
        left_eye: NDArray[np.float32],
        right_eye: NDArray[np.float32]
    ) -> Tuple[HemifieldData, HemifieldData]:
        """Traite les données des deux yeux via le chiasma.
        
        Args:
            left_eye: Activations œil gauche (rings × sectors)
            right_eye: Activations œil droit (rings × sectors)
            
        Returns:
            Tuple (hémichamp_gauche, hémichamp_droit):
            - hémichamp_gauche: champ visuel DROIT (traité par hémisphère gauche)
            - hémichamp_droit: champ visuel GAUCHE (traité par hémisphère droit)
        """
        self._frame_count += 1
        
        # Extraire les hémichamps de chaque œil
        # Œil gauche
        left_temporal = left_eye[:, self.left_temporal_sectors]   # Va vers hémisphère GAUCHE
        left_nasal = left_eye[:, self.left_nasal_sectors]         # Croise vers hémisphère DROIT
        
        # Œil droit
        right_temporal = right_eye[:, self.right_temporal_sectors] # Va vers hémisphère DROIT
        right_nasal = right_eye[:, self.right_nasal_sectors]       # Croise vers hémisphère GAUCHE
        
        # === HÉMISPHÈRE GAUCHE (traite le champ visuel DROIT) ===
        # Reçoit: temporal droit + nasal gauche
        # Note: il faut inverser les secteurs nasaux pour l'alignement spatial
        left_nasal_flipped = np.flip(left_nasal, axis=1)
        
        left_hemi = self._create_hemifield(
            temporal=right_nasal,        # Nasal droit → hémisphère gauche
            nasal=left_temporal,         # Temporal gauche → hémisphère gauche (ERREUR CORRIGÉE)
            side='left'
        )
        
        # === HÉMISPHÈRE DROIT (traite le champ visuel GAUCHE) ===
        # Reçoit: temporal gauche + nasal droit
        right_nasal_flipped = np.flip(right_nasal, axis=1)
        
        right_hemi = self._create_hemifield(
            temporal=left_nasal,         # Nasal gauche → hémisphère droit  
            nasal=right_temporal,        # Temporal droit → hémisphère droit (ERREUR CORRIGÉE)
            side='right'
        )
        
        # CORRECTION: La logique était inversée. Reprenons:
        # Hémisphère GAUCHE reçoit:
        #   - Temporal de l'œil GAUCHE (ipsilatéral) 
        #   - Nasal de l'œil DROIT (controlatéral, croise)
        #   → Voit le champ visuel DROIT
        
        # Hémisphère DROIT reçoit:
        #   - Temporal de l'œil DROIT (ipsilatéral)
        #   - Nasal de l'œil GAUCHE (controlatéral, croise)
        #   → Voit le champ visuel GAUCHE
        
        left_hemi = self._create_hemifield(
            temporal=left_temporal,      # Temporal gauche → hémisphère gauche (ipsi)
            nasal=right_nasal,           # Nasal droit → hémisphère gauche (croise)
            side='left'
        )
        
        right_hemi = self._create_hemifield(
            temporal=right_temporal,     # Temporal droit → hémisphère droit (ipsi)
            nasal=left_nasal,            # Nasal gauche → hémisphère droit (croise)
            side='right'
        )
        
        return left_hemi, right_hemi
    
    def _create_hemifield(
        self,
        temporal: NDArray[np.float32],
        nasal: NDArray[np.float32],
        side: str
    ) -> HemifieldData:
        """Crée les données d'un hémichamp à partir des composantes.
        
        Args:
            temporal: Données de l'œil ipsilatéral (partie temporale)
            nasal: Données de l'œil controlatéral (partie nasale, après croisement)
            side: 'left' ou 'right' (hémisphère receveur)
            
        Returns:
            HemifieldData avec fusion et métriques
        """
        # Les deux composantes doivent être alignées spatialement
        # Le nasal doit être inversé (miroir) pour correspondre au temporal
        nasal_aligned = np.flip(nasal, axis=1)
        
        # Calculer la disparité (différence entre les deux yeux)
        # Une disparité positive = objet plus proche
        # Une disparité négative = objet plus loin
        disparity = temporal - nasal_aligned
        
        # Calculer la corrélation locale (qualité de la correspondance)
        # Haute corrélation = même objet vu par les deux yeux
        correlation = self._compute_local_correlation(temporal, nasal_aligned)
        
        # Fusion pondérée par la corrélation
        # Si bonne corrélation: moyenne des deux
        # Si mauvaise corrélation: privilégier temporal (plus fiable)
        alpha = np.clip(correlation, 0.3, 1.0)  # Minimum 30% du nasal
        fused = alpha * temporal + (1 - alpha) * nasal_aligned
        
        # Appliquer les poids bilatéraux fovéaux si activé
        if self.config.foveal_bilateral:
            fused = self._apply_bilateral_fusion(fused, temporal, nasal_aligned)
        
        # Mettre à jour les statistiques
        self._total_disparity += np.abs(disparity).mean()
        self._total_correlation += correlation.mean()
        
        return HemifieldData(
            data=fused,
            temporal=temporal,
            nasal=nasal_aligned,
            disparity=disparity,
            correlation=correlation
        )
    
    def _compute_local_correlation(
        self,
        a: NDArray[np.float32],
        b: NDArray[np.float32],
        window: int = 3
    ) -> NDArray[np.float32]:
        """Calcule la corrélation locale entre deux images.
        
        Args:
            a, b: Images à comparer
            window: Taille de la fenêtre de corrélation
            
        Returns:
            Carte de corrélation (0 = pas de correspondance, 1 = identique)
        """
        from scipy import ndimage
        
        # Normaliser
        a_norm = (a - a.mean()) / (a.std() + 1e-8)
        b_norm = (b - b.mean()) / (b.std() + 1e-8)
        
        # Corrélation locale via convolution
        kernel = np.ones((window, window)) / (window * window)
        
        # Produit local
        ab = ndimage.convolve(a_norm * b_norm, kernel, mode='reflect')
        
        # Variances locales
        a2 = ndimage.convolve(a_norm ** 2, kernel, mode='reflect')
        b2 = ndimage.convolve(b_norm ** 2, kernel, mode='reflect')
        
        # Corrélation = covariance / (std_a * std_b)
        denom = np.sqrt(a2 * b2) + 1e-8
        correlation = np.clip(ab / denom, 0, 1)
        
        return correlation.astype(np.float32)
    
    def _apply_bilateral_fusion(
        self,
        fused: NDArray[np.float32],
        temporal: NDArray[np.float32],
        nasal: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Applique la fusion bilatérale pour les anneaux fovéaux.
        
        Dans la zone fovéale centrale, les deux yeux contribuent
        de manière plus égale (pas de division nette nasal/temporal).
        """
        result = fused.copy()
        
        for ring_idx in range(self.config.foveal_bilateral_rings):
            w = self.bilateral_weights[ring_idx]
            if w > 0:
                # Moyenne pondérée plus forte dans la fovéa
                result[ring_idx] = (temporal[ring_idx] + nasal[ring_idx]) / 2
        
        return result
    
    def get_vergence_error(
        self,
        left_hemi: HemifieldData,
        right_hemi: HemifieldData
    ) -> Tuple[float, float]:
        """Calcule l'erreur de vergence à partir des disparités.
        
        Si les yeux ne convergent pas sur le même point:
        - Disparité positive dans un hémichamp = tourner cet œil vers l'intérieur
        - Disparité négative = tourner vers l'extérieur
        
        Args:
            left_hemi: Données hémichamp gauche
            right_hemi: Données hémichamp droit
            
        Returns:
            Tuple (erreur_vergence_horizontal, erreur_vergence_vertical)
            Valeurs positives = converger plus, négatives = diverger
        """
        # Moyenne des disparités dans la zone fovéale (anneaux centraux)
        foveal_rings = self.config.foveal_bilateral_rings * 2
        
        left_disp = left_hemi.disparity[:foveal_rings].mean()
        right_disp = right_hemi.disparity[:foveal_rings].mean()
        
        # L'erreur de vergence est la moyenne des disparités
        # Si positif: les yeux doivent converger plus
        # Si négatif: les yeux doivent diverger
        h_error = (left_disp + right_disp) / 2
        
        # Pour l'erreur verticale, on regarde la différence entre haut et bas
        # (mais avec des secteurs polaires, c'est plus complexe)
        v_error = 0.0  # Simplifié pour l'instant
        
        return float(h_error), float(v_error)
    
    def get_depth_map(
        self,
        left_hemi: HemifieldData,
        right_hemi: HemifieldData
    ) -> NDArray[np.float32]:
        """Calcule une carte de profondeur à partir des disparités.
        
        Combine les disparités des deux hémichamps en une carte
        de profondeur unifiée.
        
        Args:
            left_hemi: Données hémichamp gauche (champ visuel droit)
            right_hemi: Données hémichamp droit (champ visuel gauche)
            
        Returns:
            Carte de profondeur complète (rings × sectors)
        """
        # Recombiner les deux hémichamps
        half = self.config.num_sectors // 2
        depth = np.zeros((self.config.num_rings, self.config.num_sectors), dtype=np.float32)
        
        # Hémichamp gauche → côté droit de la carte
        depth[:, half:] = left_hemi.disparity
        
        # Hémichamp droit → côté gauche de la carte (inversé)
        depth[:, :half] = np.flip(right_hemi.disparity, axis=1)
        
        return depth
    
    def get_fusion_quality(
        self,
        left_hemi: HemifieldData,
        right_hemi: HemifieldData
    ) -> float:
        """Retourne la qualité globale de la fusion binoculaire.
        
        Une valeur élevée indique que les deux yeux voient
        le même contenu (bonne correspondance).
        
        Returns:
            Score de fusion entre 0 et 1
        """
        left_corr = left_hemi.correlation.mean()
        right_corr = right_hemi.correlation.mean()
        return float((left_corr + right_corr) / 2)
    
    @property
    def stats(self) -> Dict:
        """Statistiques du chiasma."""
        if self._frame_count == 0:
            return {'frames': 0, 'avg_disparity': 0, 'avg_correlation': 0}
        
        return {
            'frames': self._frame_count,
            'avg_disparity': self._total_disparity / self._frame_count,
            'avg_correlation': self._total_correlation / self._frame_count,
        }
    
    def reset_stats(self):
        """Réinitialise les statistiques."""
        self._frame_count = 0
        self._total_disparity = 0.0
        self._total_correlation = 0.0
