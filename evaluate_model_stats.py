import time
import numpy as np
from stable_baselines3 import SAC
from env_passenv_grid import PassEnvGrid

def main():
    print("Loading environment and model...")
    # Matches original env initialization config
    env = PassEnvGrid(grid_n=16, grid_samples=3)
    
    try:
        model = SAC.load("pass_planner_sac_grid")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
        
    num_scenarios = 300
    
    success_count = 0
    inference_times_ms = []
    passing_angles_deg = []
    rewards = []
    
    print(f"Running {num_scenarios} test scenarios...")
    
    for i in range(num_scenarios):
        obs, info = env.reset()
        
        # Dummy prediction on first loop to warm-up CPU/GPU and not bias inference time
        if i == 0:
            model.predict(obs, deterministic=True)
            
        # Measure inference time
        t0 = time.time()
        action, _ = model.predict(obs, deterministic=True)
        t1 = time.time()
        
        inf_time_ms = (t1 - t0) * 1000.0
        inference_times_ms.append(inf_time_ms)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record stats
        rewards.append(reward)
        if info["all_inside"]:
            success_count += 1
            
        passing_angles_deg.append(abs(np.degrees(info["theta"])))

    # Convert to Numpy for easy calculation
    inference_times_ms = np.array(inference_times_ms)
    passing_angles_deg = np.array(passing_angles_deg)
    rewards = np.array(rewards)
    
    # Compute metrics exactly matching the paper/table
    success_rate = (success_count / num_scenarios) * 100
    mean_inf_time = np.mean(inference_times_ms)
    p95_inf_time = np.percentile(inference_times_ms, 95)
    
    median_angle = np.median(passing_angles_deg)
    p95_angle = np.percentile(passing_angles_deg, 95)
    
    median_reward = np.median(rewards)
    iqr_reward = np.percentile(rewards, 75) - np.percentile(rewards, 25)
    
    # Print the table nicely
    print("\n" + "="*60)
    print(" OVERALL PERFORMANCE OF THE PROPOSED PASSING-CONFIGURATION")
    print(" OPTIMIZATION METHOD ")
    print("="*60)
    print(f"{'Metric':<45} {'Value':<10}")
    print("-" * 60)
    print(f"{'Number of test scenarios':<45} {num_scenarios}")
    print(f"{'Success rate (%)':<45} {success_rate:.2f}")
    print(f"{'Mean inference time (ms)':<45} {mean_inf_time:.2f}")
    print(f"{'95th percentile inference time (ms)':<45} {p95_inf_time:.2f}")
    print(f"{'Median absolute passing angle |θ| (deg)':<45} {median_angle:.1f}")
    print(f"{'95th percentile |θ| (deg)':<45} {p95_angle:.1f}")
    print(f"{'Median reward':<45} {median_reward:.2f}")
    print(f"{'Reward interquartile range (P75-P25)':<45} {iqr_reward:.2f}")
    print("="*60)

if __name__ == "__main__":
    main()
