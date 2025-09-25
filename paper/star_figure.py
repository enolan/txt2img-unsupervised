import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set up the figure with three subplots
fig = plt.figure(figsize=(15, 5))

# Define colors for consistency
sphere_color = '#E8E8E8'
cap_color = '#FFE4B5'
vector1_color = '#2E86AB'
vector2_color = '#A23B72'

# First vector (reference vector)
v1 = np.array([0, 0, 1])  # pointing up

# Second vector (inside the cap) - adjust angle as needed
theta_offset = np.pi/8  # angle from v1
phi_offset = np.pi/4
v2 = np.array([
    np.sin(theta_offset) * np.cos(phi_offset),
    np.sin(theta_offset) * np.sin(phi_offset),
    np.cos(theta_offset)
])

# Spherical cap parameters
cap_angle = np.pi/6  # half-angle of the cap (adjust as needed)

# ==================
# Subplot 1: Sphere with unit vector
# ==================
ax1 = fig.add_subplot(131, projection='3d')

# Create sphere
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 20)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

ax1.plot_surface(x, y, z, alpha=0.3, color=sphere_color, linewidth=0)

# Plot first unit vector
ax1.quiver(0, 0, 0, v1[0], v1[1], v1[2], 
           color=vector1_color, arrow_length_ratio=0.15, linewidth=3)

# Add label
ax1.text(v1[0]*1.2, v1[1]*1.2, v1[2]*1.2, r'$\mathbf{v}_1$', fontsize=12)

ax1.set_title('Compute CLIP Text Embedding: "Barack Obama"', fontsize=12)
ax1.set_box_aspect([1,1,1])
ax1.set_xlim([-1.2, 1.2])
ax1.set_ylim([-1.2, 1.2])
ax1.set_zlim([-1.2, 1.2])
ax1.grid(False)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])

# ==================
# Subplot 2: Sphere with spherical cap and both vectors
# ==================
ax2 = fig.add_subplot(132, projection='3d')

# Create full sphere (lighter/transparent)
ax2.plot_surface(x, y, z, alpha=0.2, color=sphere_color, linewidth=0)

# Create spherical cap
# The cap is the portion of the sphere within angle cap_angle from v1
u_cap = np.linspace(0, 2 * np.pi, 30)
v_cap = np.linspace(0, cap_angle, 15)

# Transform to align cap with v1 (which points in z direction)
x_cap = np.outer(np.cos(u_cap), np.sin(v_cap))
y_cap = np.outer(np.sin(u_cap), np.sin(v_cap))
z_cap = np.outer(np.ones(np.size(u_cap)), np.cos(v_cap))

ax2.plot_surface(x_cap, y_cap, z_cap, alpha=0.6, color=cap_color, linewidth=0)

# Draw the boundary circle of the cap
theta_boundary = np.linspace(0, 2*np.pi, 50)
x_boundary = np.sin(cap_angle) * np.cos(theta_boundary)
y_boundary = np.sin(cap_angle) * np.sin(theta_boundary)
z_boundary = np.cos(cap_angle) * np.ones_like(theta_boundary)
ax2.plot(x_boundary, y_boundary, z_boundary, color='darkgray', linewidth=2)

# Plot both vectors
ax2.quiver(0, 0, 0, v1[0], v1[1], v1[2], 
           color=vector1_color, arrow_length_ratio=0.15, linewidth=3)
ax2.quiver(0, 0, 0, v2[0], v2[1], v2[2], 
           color=vector2_color, arrow_length_ratio=0.15, linewidth=3)

# Add labels
ax2.text(v1[0]*1.2, v1[1]*1.2, v1[2]*1.2, r'$\mathbf{v}_1$', fontsize=12)
ax2.text(v2[0]*1.2, v2[1]*1.2, v2[2]*1.2, r'$\mathbf{v}_2$', fontsize=12, color=vector2_color)

ax2.set_title('Generate Nearby Image Embedding', fontsize=12)
ax2.set_box_aspect([1,1,1])
ax2.set_xlim([-1.2, 1.2])
ax2.set_ylim([-1.2, 1.2])
ax2.set_zlim([-1.2, 1.2])
ax2.grid(False)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])

# ==================
# Subplot 3: Second vector alone
# ==================
ax3 = fig.add_subplot(133, projection='3d')

# Just plot the second vector
ax3.quiver(0, 0, 0, v2[0], v2[1], v2[2], 
           color=vector2_color, arrow_length_ratio=0.15, linewidth=3)

# Add label
ax3.text(v2[0]*1.2, v2[1]*1.2, v2[2]*1.2, r'$\mathbf{v}_2$', fontsize=12, color=vector2_color)

# Optionally add a faint unit sphere for reference
ax3.plot_surface(x, y, z, alpha=0.1, color=sphere_color, linewidth=0)

ax3.set_title('Selected Vector', fontsize=12)
ax3.set_box_aspect([1,1,1])
ax3.set_xlim([-1.2, 1.2])
ax3.set_ylim([-1.2, 1.2])
ax3.set_zlim([-1.2, 1.2])
ax3.grid(False)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_zticks([])

# Adjust layout and save
plt.tight_layout()

# Save in high resolution for paper
plt.savefig('sphere_diagram.pdf', dpi=300, bbox_inches='tight')
plt.savefig('sphere_diagram.png', dpi=300, bbox_inches='tight')

plt.show()

# Print angle between vectors for reference
angle = np.arccos(np.dot(v1, v2)) * 180 / np.pi
print(f"Angle between v1 and v2: {angle:.2f} degrees")
print(f"Spherical cap half-angle: {cap_angle * 180 / np.pi:.2f} degrees")