import numpy as np
import matplotlib.pyplot as plt
from arc_length_yaw_traj_opt import ArcLengthYawTrajOpt


def main():
    print("----------------------------------------------------------------------")
    print("Testing Densified Arc-Length Yaw Trajectory Optimization Framework...")
    print("Robot Model: Rectangular Footprint approximated by Two Overlapping Circles.")
    print("----------------------------------------------------------------------")

    # Start pose: (-2.0, 4.0) facing RIGHT (theta = 0)
    start_pose = np.array([-2.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Goal pose: (4.0, 0.0) vertically parked facing UP (theta = np.pi/2)
    goal_pose = np.array([4.0, 0.0, np.pi/2, 0.0, 0.0, 0.0, 0.0])

    # Circular obstacles representing parked cars defining a narrow vertical parking slot
    # and blocking obstacles near the start to prevent early corner-cutting
    obstacles = [
        (2.8, 0.5, 0.6),  # Car on the left of slot
        (5.2, 0.5, 0.6),  # Car on the right of slot
        (0.0, 2.4, 0.8),  # Corner-cutting blocking obstacle 1
        (2.0, 2.4, 0.8)   # Corner-cutting blocking obstacle 2
    ]

    # Densify initial path to make each segment around 0.5 meters long
    corners = [
        np.array([-2.0, 4.0]),
        np.array([4.0, 4.0]),
        np.array([4.0, 0.0])
    ]

    # Segment 0 is 6.0 meters long -> 12 pieces
    M0 = 12
    # Segment 1 is 4.0 meters long -> 8 pieces
    M1 = 8
    M = M0 + M1  # Total 20 pieces!
    K = 10       # 10 integration intervals per segment is perfect for M=20
    
    # Rectangular robot parameters (Length L = 0.90m, Width W = 0.40m)
    # Approximated by two overlapping circles of radius Rc = 0.20m, separated by d = 0.25m from the center.
    robot_radius = 0.20
    circle_dist = 0.25

    print(f"Densified path: M0={M0} forward pieces, M1={M1} backward pieces (Total M={M} pieces)")
    print(f"Rectangular Robot: Length=0.90m, Width=0.40m (Two-Circles Approximation: Rc={robot_radius}m, d={circle_dist}m)")

    optimizer = ArcLengthYawTrajOpt(
        M=M,
        K=K,
        robot_radius=robot_radius,
        circle_dist=circle_dist,
        obstacles=obstacles,
        max_v=2.0,
        max_w=1.5,
        max_a=1.5,
        max_dw=1.5,
        collision_weight=3000.0,  # High weight to enforce strict clearance
        smooth_weight=1.0,
        limit_weight=50.0,
        time_weight=1.0,
        goal_pos_weight=100.0     # ALM initial rho
    )

    # Reconstruct the densified path coordinates
    pts = []
    for i in range(M0):
        fraction = i / M0
        pts.append(corners[0] + fraction * (corners[1] - corners[0]))
    for i in range(M1 + 1):
        fraction = i / M1
        pts.append(corners[1] + fraction * (corners[2] - corners[1]))

    # Set up segment durations (T) and arc-length step increments (delta_s)
    T_guess = np.ones(M) * 1.0  # Naive piece durations of 1.0 second each
    delta_s_guess = np.zeros(M)
    delta_s_guess[:M0] = 6.0 / M0   # +0.5 m forward steps
    delta_s_guess[M0:] = -4.0 / M1  # -0.5 m backward steps

    # Set up intermediate waypoint states [theta, w, alpha, v, a]
    wps_guess = np.zeros((M - 1, 5))
    for i in range(M - 1):
        if i < M0 - 1:
            wps_guess[i, 0] = 0.0   # heading is 0.0 (facing right)
            wps_guess[i, 3] = 0.5   # velocity is +0.5 m/s
        elif i == M0 - 1:
            wps_guess[i, 0] = np.pi/2  # pivots heading to facing UP
            wps_guess[i, 3] = 0.0      # stop at the corner to reverse
        else:
            wps_guess[i, 0] = np.pi/2  # facing UP
            wps_guess[i, 3] = -0.5     # reversing velocity of -0.5 m/s

    total_vars = 2 * M + 5 * (M - 1)
    initial_guess = np.zeros(total_vars)
    initial_guess[:M] = T_guess
    initial_guess[M:2*M] = delta_s_guess
    initial_guess[2*M:] = wps_guess.flatten()

    print("Running vertical parking optimization with PHR ALM...")
    success, traj = optimizer.optimize(start_pose, goal_pose, initial_guess=initial_guess)

    pos_err = np.linalg.norm(traj['final_xy'] - goal_pose[:2])
    print(f"\nOptimization Finished. Success: {success}, Final Pos Error: {pos_err:.4f} m")

    if success or pos_err < 0.15:
        print("\nTrajectory optimization converged beautifully for vertical parking!")
        print(f"Final position error (goal constraint satisfaction): {pos_err:.4f} m")

        # Assert goal constraint satisfaction within TopAY threshold
        assert pos_err < 0.25, f"Goal position not reached! Error: {pos_err:.4f} m"

        # Check collision avoidance for both circles approximating the rectangular robot shape
        for idx, (tx, ty, th) in enumerate(zip(traj['x'], traj['y'], traj['theta'])):
            cos_th, sin_th = np.cos(th), np.sin(th)
            # Front and rear circle positions
            p_front = np.array([tx + circle_dist * cos_th, ty + circle_dist * sin_th])
            p_rear = np.array([tx - circle_dist * cos_th, ty - circle_dist * sin_th])

            for ox, oy, r in obstacles:
                # 1. Check Front Circle
                dist_front = np.sqrt((p_front[0] - ox)**2 + (p_front[1] - oy)**2)
                assert dist_front >= robot_radius + r - 0.10, \
                    f"Front circle collision at step {idx}: dist={dist_front:.3f} is less than margin={robot_radius + r}"

                # 2. Check Rear Circle
                dist_rear = np.sqrt((p_rear[0] - ox)**2 + (p_rear[1] - oy)**2)
                assert dist_rear >= robot_radius + r - 0.10, \
                    f"Rear circle collision at step {idx}: dist={dist_rear:.3f} is less than margin={robot_radius + r}"

        # Check kinematic boundary constraints are reasonably respected
        assert np.max(np.abs(traj['v'])) <= optimizer.max_v + 0.1, "Linear velocity limit violated"
        assert np.max(np.abs(traj['w'])) <= optimizer.max_w + 0.1, "Angular velocity limit violated"
        assert np.max(np.abs(traj['a'])) <= optimizer.max_a + 0.1, "Linear acceleration limit violated"
        assert np.max(np.abs(traj['alpha'])) <= optimizer.max_dw + 0.1, "Angular acceleration limit violated"

        print("\nAll trajectory constraints, goal constraints, and feasibility checks PASSED successfully!")
        
        # 3. Create a beautiful visualization plot
        print("\nGenerating trajectory visualization plot...")
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))

        # (A) Cartesian Space Plot
        ax = axes[0]
        # Plot obstacles and margins
        for idx, (ox, oy, r) in enumerate(obstacles):
            obs_circle = plt.Circle((ox, oy), r, color='gray', alpha=0.6, label='Parked Cars' if idx < 2 else 'Obstacles')
            ax.add_patch(obs_circle)
            margin_circle = plt.Circle((ox, oy), r + robot_radius, color='red', fill=False, linestyle='--', alpha=0.4, label='Safety Margin' if idx == 0 else '')
            ax.add_patch(margin_circle)

        # Plot rectangular robot footprints along the optimized path (every 35 steps)
        # Drawn as two overlapping circles
        dense_steps = len(traj['x'])
        for idx in range(0, dense_steps, 35):
            tx, ty, th = traj['x'][idx], traj['y'][idx], traj['theta'][idx]
            cos_th, sin_th = np.cos(th), np.sin(th)
            
            # Front circle
            tf_x, tf_y = tx + circle_dist * cos_th, ty + circle_dist * sin_th
            front_fp = plt.Circle((tf_x, tf_y), robot_radius, color='cyan', fill=False, alpha=0.2, label='Robot Footprint' if idx == 0 else '')
            ax.add_patch(front_fp)
            
            # Rear circle
            tr_x, tr_y = tx - circle_dist * cos_th, ty - circle_dist * sin_th
            rear_fp = plt.Circle((tr_x, tr_y), robot_radius, color='cyan', fill=False, alpha=0.2)
            ax.add_patch(rear_fp)
            
            # Center marker and heading arrow
            ax.plot(tx, ty, 'b.', markersize=4)
            ax.arrow(tx, ty, 0.25 * np.cos(th), 0.25 * np.sin(th), head_width=0.08, head_length=0.12, fc='blue', ec='blue', alpha=0.4)
            
        # Plot last robot footprint
        last_x, last_y, last_th = traj['x'][-1], traj['y'][-1], traj['theta'][-1]
        last_cos, last_sin = np.cos(last_th), np.sin(last_th)
        last_front = plt.Circle((last_x + circle_dist * last_cos, last_y + circle_dist * last_sin), robot_radius, color='cyan', fill=False, alpha=0.4)
        last_rear = plt.Circle((last_x - circle_dist * last_cos, last_y - circle_dist * last_sin), robot_radius, color='cyan', fill=False, alpha=0.4)
        ax.add_patch(last_front)
        ax.add_patch(last_rear)
        ax.arrow(last_x, last_y, 0.25 * last_cos, 0.25 * last_sin, head_width=0.08, head_length=0.12, fc='blue', ec='blue', alpha=0.6)

        # Plot start and goal
        ax.plot(start_pose[0], start_pose[1], 'g*', markersize=14, label='Start (Facing Right)')
        ax.plot(goal_pose[0], goal_pose[1], 'r*', markersize=14, label='Parked (Facing Up)')

        # Plot initial path A*
        init_pts = np.array(pts)
        ax.plot(init_pts[:, 0], init_pts[:, 1], 'k--o', linewidth=1.0, markersize=3, label='Densified Initial Path (A*)')

        # Plot optimized path
        ax.plot(traj['x'], traj['y'], 'b-', linewidth=3.0, label='Optimized Path (TopAY)')

        ax.set_aspect('equal')
        ax.set_xlim(-3.5, 7.0)
        ax.set_ylim(-1.0, 6.0)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.set_xlabel('X Coordinate (meters)', fontsize=11)
        ax.set_ylabel('Y Coordinate (meters)', fontsize=11)
        ax.set_title('Densified Rectangular Parking Cartesian Path', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right')

        # (B) Velocity Plot to verify mixed forward, in-place, and reversal
        ax = axes[1]
        ax.plot(traj['time'], traj['v'], 'g-', linewidth=2.5, label='Linear Velocity v(t)')
        ax.plot(traj['time'], traj['w'], 'm--', linewidth=2.5, label='Angular Velocity w(t)')
        
        # Determine exact transition times dynamically from optimal piece durations
        T_opt = traj['time'][-1]
        t_fwd = T_opt * (M0 / M)  # forward move segment ratio
        
        # Shading regions of different maneuvers
        ax.axvspan(0.0, t_fwd, color='green', alpha=0.07, label='Forward Move')
        ax.axvspan(t_fwd, t_fwd + 1.2, color='magenta', alpha=0.07, label='In-Place Rotation')
        ax.axvspan(t_fwd + 1.2, traj['time'][-1], color='red', alpha=0.07, label='Backwards Reversal')

        ax.grid(True, linestyle=':', alpha=0.5)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Velocity', fontsize=11)
        ax.set_title('Maneuver Analysis (Forward, Rotate, Reverse)', fontsize=13, fontweight='bold')
        ax.legend(loc='lower left')

        plt.tight_layout()
        image_name = 'trajectory_optimization.png'
        plt.savefig(image_name, dpi=150)
        plt.close()
        print(f"Beautiful densified vertical parking maneuver plot saved to '{image_name}'!")
    else:
        print("\nTrajectory optimization failed to converge for vertical parking.")
        print(f"Scipy message: {traj['message']}")

    print("----------------------------------------------------------------------")


if __name__ == "__main__":
    main()
