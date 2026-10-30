import numpy as np
import matplotlib.pyplot as plt
from numba import njit

def generate_regular_polygon(n_sides=5, radius=1.0):
    """Generate vertices of a regular n-gon centered at the origin."""
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    if n_sides % 2 == 1:
        angles += np.pi / n_sides
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

@njit
def get_outer_billiard_vertex(point, vertices):
    """Numba-compiled vertex lookup for dual billiards reflection."""
    n = len(vertices)
    for i in range(n):
        v_prev = vertices[i - 1]
        v = vertices[i]
        v_next = vertices[(i + 1) % n]
        
        edge1 = v - v_prev
        to_pt1 = point - v_prev
        cp1 = edge1[0] * to_pt1[1] - edge1[1] * to_pt1[0]
        
        edge2 = v_next - v
        to_pt2 = point - v
        cp2 = edge2[0] * to_pt2[1] - edge2[1] * to_pt2[0]
        
        if cp1 >= 0 and cp2 < 0:
            return v
            
    # Fallback nearest vertex using explicit loop for Numba compatibility
    min_dist = np.inf
    best_v = vertices[0]
    for i in range(n):
        v = vertices[i]
        dist = np.sqrt((point[0] - v[0])**2 + (point[1] - v[1])**2)
        if dist < min_dist:
            min_dist = dist
            best_v = v
    return best_v

@njit
def compute_orbit_points(xs, ys, vertices, iterations, spatial_extent):
    """
    Runs trajectories over the grid and writes directly to a pre-allocated 
    flat array for maximum speed.
    """
    max_total_points = len(xs) * len(ys) * iterations
    # Pre-allocate flat arrays to bypass list.append() limitations in Numba
    coords_x = np.empty(max_total_points, dtype=np.float64)
    coords_y = np.empty(max_total_points, dtype=np.float64)
    
    count = 0
    for i in range(len(xs)):
        for j in range(len(ys)):
            pt = np.array([xs[i], ys[j]], dtype=np.float64)
            
            # Skip points inside or too close to the table
            if np.sqrt(pt[0]**2 + pt[1]**2) < 1.15:
                continue
                
            for _ in range(iterations):
                v_ref = get_outer_billiard_vertex(pt, vertices)
                # Reflection formula: P' = 2*V - P
                pt[0] = 2.0 * v_ref[0] - pt[0]
                pt[1] = 2.0 * v_ref[1] - pt[1]
                
                # Prune points escaping the view boundaries
                if np.sqrt(pt[0]**2 + pt[1]**2) > spatial_extent * 2.5:
                    break
                    
                coords_x[count] = pt[0]
                coords_y[count] = pt[1]
                count += 1
                
    return coords_x[:count], coords_y[:count]

def run_dual_billiards_simulation(n_sides=5, grid_density=150, iterations=200, spatial_extent=3.5):
    """Standard Python wrapper handling setup and plotting."""
    vertices = generate_regular_polygon(n_sides=n_sides, radius=1.0)
    
    xs = np.linspace(-spatial_extent, spatial_extent, grid_density)
    ys = np.linspace(-spatial_extent, spatial_extent, grid_density)
    
    print(f"Computing dual billiards for a {n_sides}-gon...")
    all_x, all_y = compute_orbit_points(xs, ys, vertices, iterations, spatial_extent)
    print(f"Done! Plotting {len(all_x):,} total points...")

    # Plotting configuration
    plt.figure(figsize=(10, 10), dpi=150)
    plt.scatter(all_x, all_y, marker='.', s=1, c='black', alpha=0.25)
    
    # Draw polygon table
    poly_closed = np.vstack([vertices, vertices[0]])
    plt.plot(poly_closed[:, 0], poly_closed[:, 1], 'r-', linewidth=2, label=f"{n_sides}-gon Table")
    
    plt.title(f"Outer Billiards Phase Space ({n_sides}-sided Table)")
    plt.gca().set_aspect('equal')
    plt.xlim(-spatial_extent, spatial_extent)
    plt.ylim(-spatial_extent, spatial_extent)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# --- EXAMPLE USAGE ---
run_dual_billiards_simulation(n_sides=4, grid_density=100, iterations=250, spatial_extent=4.0)
