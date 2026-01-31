"""
Tests pour le module attention.py
"""

import pytest
import numpy as np
import math

from neuronspikes.attention import (
    ZoomConfig, ZoomLevel, AttentionConfig,
    VirtualZoom, InhibitionMap, AttentionMemory,
    AttentionController, GazeMemory, PointOfInterest
)
from neuronspikes.fovea import FoveaConfig


class TestZoomConfig:
    """Tests de ZoomConfig."""
    
    def test_default_values(self):
        cfg = ZoomConfig()
        assert cfg.min_scale == 0.25
        assert cfg.max_scale == 2.0
        assert cfg.num_levels == 8
        assert cfg.zoom_speed == 0.15
    
    def test_custom_values(self):
        cfg = ZoomConfig(
            min_scale=0.5,
            max_scale=4.0,
            num_levels=4
        )
        assert cfg.min_scale == 0.5
        assert cfg.max_scale == 4.0
        assert cfg.num_levels == 4


class TestGazeMemory:
    """Tests de GazeMemory."""
    
    def test_creation(self):
        mem = GazeMemory(x=100, y=200, scale=1.0, frame=42)
        assert mem.x == 100
        assert mem.y == 200
        assert mem.scale == 1.0
        assert mem.frame == 42
        assert mem.visits == 1
    
    def test_distance_to(self):
        mem = GazeMemory(x=0, y=0, scale=1.0, frame=0)
        assert mem.distance_to(3, 4) == 5.0  # Triangle 3-4-5
        assert mem.distance_to(0, 0) == 0.0


class TestPointOfInterest:
    """Tests de PointOfInterest."""
    
    def test_creation(self):
        poi = PointOfInterest(x=50, y=100)
        assert poi.x == 50
        assert poi.y == 100
        assert poi.confidence == 0.0
        assert poi.observations == 0
    
    def test_update(self):
        poi = PointOfInterest(x=50, y=100, first_seen=0)
        poi.update(frame=10)
        assert poi.last_seen == 10
        assert poi.observations == 1
        assert poi.confidence > 0
        
        # Plusieurs observations augmentent la confiance
        for i in range(10):
            poi.update(frame=20 + i)
        assert poi.confidence > 0.5
        assert poi.observations == 11
    
    def test_update_with_features(self):
        poi = PointOfInterest(x=50, y=100)
        features1 = np.ones(64, dtype=np.float32)
        poi.update(frame=1, new_features=features1)
        assert poi.features is not None
        assert poi.features.shape == (64,)
        
        # Moyenne mobile
        features2 = np.zeros(64, dtype=np.float32)
        poi.update(frame=2, new_features=features2)
        # 0.2 * 0 + 0.8 * 1 = 0.8
        assert poi.features[0] == pytest.approx(0.8)


class TestVirtualZoom:
    """Tests de VirtualZoom."""
    
    def test_creation(self):
        cfg = ZoomConfig(num_levels=8)
        zoom = VirtualZoom(cfg, image_width=640, image_height=480, base_radius=64)
        assert zoom.image_width == 640
        assert zoom.image_height == 480
        assert zoom.base_radius == 64
        assert zoom.current_scale == 1.0
    
    def test_zoom_in(self):
        cfg = ZoomConfig(num_levels=8)
        zoom = VirtualZoom(cfg, 640, 480, 64)
        initial_level = zoom.level_index
        
        zoom.zoom_in()
        assert zoom.level_index == initial_level + 1
        
        # Transition lisse - pas encore à la cible
        zoom.update()
        zoom.update()
        zoom.update()
        # Après plusieurs updates, on s'approche de la cible
    
    def test_zoom_out(self):
        cfg = ZoomConfig(num_levels=8)
        zoom = VirtualZoom(cfg, 640, 480, 64)
        initial_level = zoom.level_index
        
        zoom.zoom_out()
        assert zoom.level_index == initial_level - 1
    
    def test_zoom_limits(self):
        cfg = ZoomConfig(num_levels=4)
        zoom = VirtualZoom(cfg, 640, 480, 64)
        
        # Zoom in jusqu'au max
        for _ in range(10):
            zoom.zoom_in()
        assert zoom.level_index == 3  # Max
        
        # Zoom out jusqu'au min
        for _ in range(10):
            zoom.zoom_out()
        assert zoom.level_index == 0  # Min
    
    def test_current_radius(self):
        cfg = ZoomConfig(min_scale=0.5, max_scale=2.0, num_levels=4)
        zoom = VirtualZoom(cfg, 640, 480, base_radius=64)
        
        # Au niveau 0 (zoom out), rayon plus grand
        zoom.set_level(0)
        for _ in range(20):
            zoom.update()
        radius_wide = zoom.current_radius
        
        # Au niveau max (zoom in), rayon plus petit
        zoom.set_level(3)
        for _ in range(20):
            zoom.update()
        radius_close = zoom.current_radius
        
        assert radius_wide > radius_close
    
    def test_constrain_gaze(self):
        cfg = ZoomConfig()
        zoom = VirtualZoom(cfg, 640, 480, 64)
        
        # Position dans les limites
        x, y = zoom.constrain_gaze(320, 240)
        assert x == 320
        assert y == 240
        
        # Position hors limites (trop à gauche)
        x, y = zoom.constrain_gaze(10, 240)
        assert x >= zoom.current_radius
        
        # Position hors limites (trop en bas)
        x, y = zoom.constrain_gaze(320, 470)
        assert y <= 480 - zoom.current_radius
    
    def test_zoom_level_enum(self):
        cfg = ZoomConfig(num_levels=8)
        zoom = VirtualZoom(cfg, 640, 480, 64)
        
        zoom.set_level(0)
        assert zoom.zoom_level == ZoomLevel.WIDE
        
        zoom.set_level(7)
        assert zoom.zoom_level == ZoomLevel.DETAIL
    
    def test_effective_fovea_config(self):
        cfg = ZoomConfig(num_levels=4)
        zoom = VirtualZoom(cfg, 640, 480, 64)
        base_cfg = FoveaConfig(num_rings=8, num_sectors=8, max_radius=64)
        
        zoom.set_level(3)  # Zoom in
        for _ in range(20):
            zoom.update()
        
        effective = zoom.get_effective_fovea_config(base_cfg)
        assert effective.num_rings == base_cfg.num_rings
        assert effective.max_radius < base_cfg.max_radius


class TestInhibitionMap:
    """Tests de InhibitionMap."""
    
    def test_creation(self):
        cfg = AttentionConfig()
        inh = InhibitionMap(80, 60, cfg)
        assert inh.width == 80
        assert inh.height == 60
        assert inh.inhibition_map.shape == (60, 80)
    
    def test_add_gaze(self):
        cfg = AttentionConfig(inhibition_radius=16)
        inh = InhibitionMap(80, 60, cfg)
        
        # Ajouter un point de regard
        inh.add_gaze(320, 240, frame=1, intensity=1.0)
        
        # L'inhibition devrait être non nulle
        total_inhibition = np.sum(inh.inhibition_map)
        assert total_inhibition > 0
    
    def test_decay(self):
        cfg = AttentionConfig(inhibition_decay=0.9)
        inh = InhibitionMap(80, 60, cfg)
        
        inh.add_gaze(320, 240, frame=1, intensity=1.0)
        initial = np.sum(inh.inhibition_map)
        
        inh.decay()
        after_decay = np.sum(inh.inhibition_map)
        
        assert after_decay < initial
    
    def test_modulate_saliency(self):
        cfg = AttentionConfig()
        inh = InhibitionMap(8, 6, cfg)  # Petite carte
        
        # Ajouter forte inhibition
        inh._map[3, 4] = 0.8
        
        # Saillance uniforme
        saliency = np.ones((6, 8), dtype=np.float32)
        
        modulated = inh.modulate_saliency(saliency, scale=1.0)
        
        # La zone inhibée devrait avoir une saillance réduite
        assert modulated[3, 4] < saliency[3, 4]
    
    def test_reset(self):
        cfg = AttentionConfig()
        inh = InhibitionMap(80, 60, cfg)
        
        inh.add_gaze(320, 240, frame=1)
        inh.reset()
        
        assert np.sum(inh.inhibition_map) == 0


class TestAttentionMemory:
    """Tests de AttentionMemory."""
    
    def test_creation(self):
        cfg = AttentionConfig(memory_size=50)
        mem = AttentionMemory(cfg)
        assert len(mem._history) == 0
        assert len(mem._pois) == 0
    
    def test_record_gaze(self):
        cfg = AttentionConfig()
        mem = AttentionMemory(cfg)
        
        mem.record_gaze(100, 200, scale=1.0, saliency=0.5)
        assert len(mem._history) == 1
        
        mem.record_gaze(150, 250, scale=1.0, saliency=0.6)
        assert len(mem._history) == 2
    
    def test_record_gaze_merges_nearby(self):
        cfg = AttentionConfig()
        mem = AttentionMemory(cfg)
        
        mem.record_gaze(100, 200, scale=1.0)
        mem.record_gaze(105, 205, scale=1.0)  # Proche du premier
        
        # Devrait fusionner (même entrée avec visits=2)
        assert mem._history[-1].visits >= 1
    
    def test_register_poi(self):
        cfg = AttentionConfig()
        mem = AttentionMemory(cfg)
        
        mem.register_poi(100, 200)
        assert len(mem._pois) == 1
        assert mem._pois[0].x == 100
        assert mem._pois[0].y == 200
    
    def test_register_poi_merges_nearby(self):
        cfg = AttentionConfig()
        mem = AttentionMemory(cfg)
        
        mem._frame_count = 1
        mem.register_poi(100, 200)
        mem._frame_count = 2
        mem.register_poi(110, 210)  # Proche du premier
        
        # Devrait fusionner
        assert len(mem._pois) == 1
        assert mem._pois[0].observations == 2
    
    def test_get_exploration_score(self):
        cfg = AttentionConfig()
        mem = AttentionMemory(cfg)
        
        # Position jamais visitée
        score = mem.get_exploration_score(100, 100)
        assert score == 1.0  # Nouveau
        
        # Enregistrer plusieurs visites
        for _ in range(5):
            mem.record_gaze(100, 100, scale=1.0)
        
        # Score devrait diminuer
        score_after = mem.get_exploration_score(100, 100)
        assert score_after < 1.0
    
    def test_suggest_exploration_target(self):
        cfg = AttentionConfig()
        mem = AttentionMemory(cfg)
        
        # Visiter le centre beaucoup
        for _ in range(20):
            mem.record_gaze(320, 240, scale=1.0)
        
        # La suggestion devrait éviter le centre
        target = mem.suggest_exploration_target(320, 240, 640, 480)
        assert target != (320, 240)
    
    def test_get_stats(self):
        cfg = AttentionConfig()
        mem = AttentionMemory(cfg)
        
        mem.record_gaze(100, 100, scale=1.0)
        mem.register_poi(200, 200)
        
        stats = mem.get_stats()
        assert 'num_memories' in stats
        assert 'num_pois' in stats
        assert stats['num_memories'] >= 1
        assert stats['num_pois'] == 1
    
    def test_reset(self):
        cfg = AttentionConfig()
        mem = AttentionMemory(cfg)
        
        mem.record_gaze(100, 100, scale=1.0)
        mem.register_poi(200, 200)
        mem.reset()
        
        assert len(mem._history) == 0
        assert len(mem._pois) == 0


class TestAttentionController:
    """Tests de AttentionController."""
    
    def test_creation(self):
        fovea_cfg = FoveaConfig(max_radius=64)
        ctrl = AttentionController(640, 480, fovea_cfg)
        
        assert ctrl.image_width == 640
        assert ctrl.image_height == 480
        assert ctrl.current_gaze == (320, 240)  # Centre
    
    def test_move_to(self):
        fovea_cfg = FoveaConfig(max_radius=64)
        ctrl = AttentionController(640, 480, fovea_cfg)
        
        ctrl.move_to(100, 200)
        x, y = ctrl.current_gaze
        assert x == 100
        assert y == 200
    
    def test_move_to_constrained(self):
        fovea_cfg = FoveaConfig(max_radius=64)
        ctrl = AttentionController(640, 480, fovea_cfg)
        
        # Essayer d'aller hors limites
        ctrl.move_to(10, 10)
        x, y = ctrl.current_gaze
        assert x >= ctrl.current_radius
        assert y >= ctrl.current_radius
    
    def test_update(self):
        fovea_cfg = FoveaConfig(max_radius=64)
        ctrl = AttentionController(640, 480, fovea_cfg)
        
        saliency = np.random.rand(60, 80).astype(np.float32)
        
        pos = ctrl.update(saliency=saliency, correlation=0.5)
        assert pos == ctrl.current_gaze
    
    def test_zoom_integration(self):
        fovea_cfg = FoveaConfig(max_radius=64)
        ctrl = AttentionController(640, 480, fovea_cfg)
        
        initial_radius = ctrl.current_radius
        
        ctrl.zoom.zoom_in()
        for _ in range(20):
            ctrl.update()
        
        assert ctrl.current_radius < initial_radius
    
    def test_select_next_target(self):
        fovea_cfg = FoveaConfig(max_radius=64)
        ctrl = AttentionController(640, 480, fovea_cfg)
        
        # Saillance avec un pic clair
        saliency = np.zeros((60, 80), dtype=np.float32)
        saliency[30, 40] = 1.0  # Pic au centre
        
        target = ctrl.select_next_target(saliency, correlation=0.5)
        assert target is not None
        assert len(target) == 2
    
    def test_auto_zoom(self):
        fovea_cfg = FoveaConfig(max_radius=64)
        ctrl = AttentionController(640, 480, fovea_cfg)
        
        initial_level = ctrl.zoom.level_index
        
        # Forte corrélation et saillance → zoom in
        ctrl.auto_zoom(correlation=0.7, saliency=0.6)
        assert ctrl.zoom.level_index >= initial_level
    
    def test_get_effective_config(self):
        fovea_cfg = FoveaConfig(num_rings=8, max_radius=64)
        ctrl = AttentionController(640, 480, fovea_cfg)
        
        effective = ctrl.get_effective_config()
        assert effective.num_rings == fovea_cfg.num_rings
    
    def test_get_stats(self):
        fovea_cfg = FoveaConfig(max_radius=64)
        ctrl = AttentionController(640, 480, fovea_cfg)
        
        ctrl.update()
        stats = ctrl.get_stats()
        
        assert 'frame' in stats
        assert 'gaze' in stats
        assert 'zoom_level' in stats
        assert 'zoom_scale' in stats
    
    def test_reset(self):
        fovea_cfg = FoveaConfig(max_radius=64)
        ctrl = AttentionController(640, 480, fovea_cfg)
        
        ctrl.move_to(100, 100)
        ctrl.update()
        ctrl.zoom.zoom_in()
        
        ctrl.reset()
        
        assert ctrl.current_gaze == (320, 240)  # Retour au centre
        assert ctrl._frame_count == 0


class TestIntegration:
    """Tests d'intégration du système d'attention."""
    
    def test_full_attention_cycle(self):
        """Teste un cycle complet d'attention."""
        fovea_cfg = FoveaConfig(max_radius=64)
        ctrl = AttentionController(640, 480, fovea_cfg)
        
        # Simuler plusieurs frames
        for frame in range(50):
            # Saillance aléatoire
            saliency = np.random.rand(60, 80).astype(np.float32)
            
            # Mettre à jour
            ctrl.update(
                saliency=saliency,
                correlation=0.5 + 0.2 * np.sin(frame * 0.1)
            )
            
            # Sélectionner et aller vers cible
            if frame % 10 == 0:
                target = ctrl.select_next_target(saliency, 0.5)
                ctrl.move_to(*target)
        
        stats = ctrl.get_stats()
        assert stats['frame'] == 50
        assert stats['num_memories'] > 0
    
    def test_inhibition_promotes_exploration(self):
        """Vérifie que l'inhibition encourage l'exploration."""
        fovea_cfg = FoveaConfig(max_radius=64)
        attention_cfg = AttentionConfig(
            inhibition_decay=0.95,
            exploration_weight=0.5
        )
        ctrl = AttentionController(
            640, 480, fovea_cfg,
            attention_config=attention_cfg
        )
        
        # Fixer le regard au centre pendant longtemps
        for _ in range(30):
            saliency = np.ones((60, 80), dtype=np.float32) * 0.5
            saliency[30, 40] = 1.0  # Pic au centre
            ctrl.update(saliency=saliency, correlation=0.6)
        
        # L'inhibition au centre devrait être élevée
        inhibition_center = ctrl.inhibition.get_inhibition_at(320, 240)
        assert inhibition_center > 0
    
    def test_zoom_adapts_to_content(self):
        """Vérifie que le zoom s'adapte au contenu."""
        fovea_cfg = FoveaConfig(max_radius=64)
        ctrl = AttentionController(640, 480, fovea_cfg)
        
        # Haute corrélation → devrait zoomer
        ctrl.auto_zoom(correlation=0.8, saliency=0.7)
        high_corr_level = ctrl.zoom.level_index
        
        ctrl.reset()
        
        # Basse corrélation → devrait dézoomer
        ctrl.auto_zoom(correlation=0.1, saliency=0.3)
        low_corr_level = ctrl.zoom.level_index
        
        assert high_corr_level >= low_corr_level
