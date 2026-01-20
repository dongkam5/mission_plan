"""
RRT/RRT* 경로탐색 알고리즘
"""
import math
import random
import numpy as np
from typing import List, Tuple, Optional
from modules.config import GRID_SIZE, MAP_BOUNDS, ALTITUDE_MIN, ALTITUDE_MAX


class RRTNode:
    """RRT 노드"""
    def __init__(self, lat: float, lon: float, alt: float = 0):
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.parent: Optional[RRTNode] = None
        self.cost = 0.0  # RRT*용


class RRTPathfinder:
    """RRT 경로탐색"""
    
    def __init__(self, max_iterations: int = 5000, step_size: float = 0.1):
        self.max_iterations = max_iterations
        self.step_size = step_size  # degree 단위
        self.bounds = [
            MAP_BOUNDS["min_lat"],
            MAP_BOUNDS["max_lat"],
            MAP_BOUNDS["min_lon"],
            MAP_BOUNDS["max_lon"]
        ]
    
    def distance(self, node1: RRTNode, node2: RRTNode) -> float:
        """두 노드 간 거리 (km)"""
        return math.sqrt(
            ((node1.lat - node2.lat) * 111) ** 2 + 
            ((node1.lon - node2.lon) * 111 * math.cos(math.radians(node1.lat))) ** 2
        )
    
    def is_collision(self, node: RRTNode, threats: List[dict], margin: float) -> bool:
        """충돌 체크"""
        margin_deg = margin / 111.0
        
        for t in threats:
            if t['type'] == "SAM":
                dist_km = math.sqrt(
                    ((node.lat - t['lat']) * 111) ** 2 + 
                    ((node.lon - t['lon']) * 111 * math.cos(math.radians(node.lat))) ** 2
                )
                if dist_km < (t['radius_km'] + margin):
                    return True
            elif t['type'] == "NFZ":
                if ((t['lat_min'] - margin_deg <= node.lat <= t['lat_max'] + margin_deg) and
                    (t['lon_min'] - margin_deg <= node.lon <= t['lon_max'] + margin_deg)):
                    return True
        return False
    
    def is_path_clear(self, node1: RRTNode, node2: RRTNode, threats: List[dict], margin: float) -> bool:
        """두 노드 사이 경로가 안전한지 체크"""
        steps = 10
        for i in range(steps + 1):
            t = i / steps
            lat = node1.lat + t * (node2.lat - node1.lat)
            lon = node1.lon + t * (node2.lon - node1.lon)
            temp_node = RRTNode(lat, lon)
            if self.is_collision(temp_node, threats, margin):
                return False
        return True
    
    def sample_random_node(self, goal: RRTNode, goal_bias: float = 0.1) -> RRTNode:
        """랜덤 노드 샘플링 (goal bias)"""
        if random.random() < goal_bias:
            return goal
        
        lat = random.uniform(self.bounds[0], self.bounds[1])
        lon = random.uniform(self.bounds[2], self.bounds[3])
        return RRTNode(lat, lon)
    
    def get_nearest_node(self, tree: List[RRTNode], target: RRTNode) -> RRTNode:
        """트리에서 가장 가까운 노드 찾기"""
        return min(tree, key=lambda node: self.distance(node, target))
    
    def steer(self, from_node: RRTNode, to_node: RRTNode) -> RRTNode:
        """step_size만큼 이동한 새 노드 생성"""
        dist = self.distance(from_node, to_node)
        
        if dist <= self.step_size * 111:  # km → degree 근사
            return to_node
        
        ratio = (self.step_size * 111) / dist
        new_lat = from_node.lat + ratio * (to_node.lat - from_node.lat)
        new_lon = from_node.lon + ratio * (to_node.lon - from_node.lon)
        
        return RRTNode(new_lat, new_lon)
    
    def find_path(
        self,
        start: List[float],
        end: List[float],
        threats: List[dict],
        safety_margin: float
    ) -> List[Tuple[float, float]]:
        """RRT 경로탐색"""
        
        start_node = RRTNode(start[0], start[1])
        goal_node = RRTNode(end[0], end[1])
        
        tree = [start_node]
        
        for i in range(self.max_iterations):
            # 1. 랜덤 샘플링
            rand_node = self.sample_random_node(goal_node)
            
            # 2. 가장 가까운 노드
            nearest = self.get_nearest_node(tree, rand_node)
            
            # 3. Steer
            new_node = self.steer(nearest, rand_node)
            
            # 4. 충돌 체크
            if self.is_collision(new_node, threats, safety_margin):
                continue
            
            if not self.is_path_clear(nearest, new_node, threats, safety_margin):
                continue
            
            # 5. 트리에 추가
            new_node.parent = nearest
            tree.append(new_node)
            
            # 6. 목표 도달 체크
            if self.distance(new_node, goal_node) < 0.5:  # 0.5 degree ≈ 55km
                if self.is_path_clear(new_node, goal_node, threats, safety_margin):
                    goal_node.parent = new_node
                    tree.append(goal_node)
                    
                    # 경로 복원
                    path = []
                    current = goal_node
                    while current is not None:
                        path.append((current.lat, current.lon))
                        current = current.parent
                    
                    print(f"✅ RRT 성공: {i+1}회 반복, 경로 길이 {len(path)}")
                    return path[::-1]
        
        print("⚠️ RRT 탐색 실패 (최대 반복 도달)")
        return []


class RRTStarPathfinder(RRTPathfinder):
    """RRT* 경로탐색 (점진적 최적화)"""
    
    def __init__(self, max_iterations: int = 5000, step_size: float = 0.1, rewire_radius: float = 0.5):
        super().__init__(max_iterations, step_size)
        self.rewire_radius = rewire_radius  # degree
    
    def get_nearby_nodes(self, tree: List[RRTNode], center: RRTNode) -> List[RRTNode]:
        """반경 내 노드들 찾기"""
        return [node for node in tree if self.distance(node, center) < self.rewire_radius * 111]
    
    def find_path(
        self,
        start: List[float],
        end: List[float],
        threats: List[dict],
        safety_margin: float
    ) -> List[Tuple[float, float]]:
        """RRT* 경로탐색"""
        
        start_node = RRTNode(start[0], start[1])
        start_node.cost = 0
        goal_node = RRTNode(end[0], end[1])
        
        tree = [start_node]
        
        for i in range(self.max_iterations):
            rand_node = self.sample_random_node(goal_node)
            nearest = self.get_nearest_node(tree, rand_node)
            new_node = self.steer(nearest, rand_node)
            
            if self.is_collision(new_node, threats, safety_margin):
                continue
            
            if not self.is_path_clear(nearest, new_node, threats, safety_margin):
                continue
            
            # RRT* 개선: 최소 비용 부모 찾기
            nearby_nodes = self.get_nearby_nodes(tree, new_node)
            min_cost = nearest.cost + self.distance(nearest, new_node)
            min_node = nearest
            
            for near in nearby_nodes:
                cost = near.cost + self.distance(near, new_node)
                if cost < min_cost and self.is_path_clear(near, new_node, threats, safety_margin):
                    min_cost = cost
                    min_node = near
            
            new_node.parent = min_node
            new_node.cost = min_cost
            tree.append(new_node)
            
            # Rewiring: 주변 노드들의 부모 재설정
            for near in nearby_nodes:
                cost = new_node.cost + self.distance(new_node, near)
                if cost < near.cost and self.is_path_clear(new_node, near, threats, safety_margin):
                    near.parent = new_node
                    near.cost = cost
            
            # 목표 도달
            if self.distance(new_node, goal_node) < 0.5:
                if self.is_path_clear(new_node, goal_node, threats, safety_margin):
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + self.distance(new_node, goal_node)
                    tree.append(goal_node)
                    
                    path = []
                    current = goal_node
                    while current is not None:
                        path.append((current.lat, current.lon))
                        current = current.parent
                    
                    print(f"✅ RRT* 성공: {i+1}회 반복, 경로 길이 {len(path)}, 비용 {goal_node.cost:.2f}")
                    return path[::-1]
        
        print("⚠️ RRT* 탐색 실패")
        return []
