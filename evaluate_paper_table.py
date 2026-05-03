import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from env_passenv_grid import PassEnvGrid
import pandas as pd

def run_evaluation(model_path="pass_planner_sac_grid", num_scenarios=300):
    # Initialize the environment
    env = PassEnvGrid(grid_n=16, grid_samples=3)
    
    print(f"Loading model from {model_path}...")
    model = SAC.load(model_path, env=env)
    
    print(f"Running evaluation on {num_scenarios} scenarios...")
    
    # 统计指标记录
    success_count = 0
    inference_times = []
    passing_angles = []
    rewards = []
    margins = []
    
    for i in range(num_scenarios):
        # 强制使用完全随机的不规则多边形（通过传入 options 禁用模板）
        obs, _ = env.reset(options={"use_random": True})
        
        # Measure inference time
        start_time = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        end_time = time.perf_counter()
        
        inference_time_ms = (end_time - start_time) * 1000.0
        inference_times.append(inference_time_ms)
        
        # Step the environment
        obs2, reward, terminated, truncated, info = env.step(action)
        
        # Record basic metrics
        success_count += int(info["all_inside"])
        rewards.append(reward)
        passing_angles.append(np.abs(np.rad2deg(info["theta"]))) # Absolute passing angle |θ| in degrees
        margins.append(info["min_margin"])
        
    # === Compute Metrics for the Table ===
    success_rate = (success_count / num_scenarios) * 100.0
    
    mean_inference_time = np.mean(inference_times)
    p95_inference_time = np.percentile(inference_times, 95)
    
    median_passing_angle = np.median(passing_angles)
    p95_passing_angle = np.percentile(passing_angles, 95)
    
    median_reward = np.median(rewards)
    q75_reward, q25_reward = np.percentile(rewards, [75, 25])
    iqr_reward = q75_reward - q25_reward
    
    # 追加一些有意义的参考指标：
    mean_margin = np.mean(margins)
    min_margin_overall = np.min(margins)

    # === Print the Table ===
    print("\n=======================================================")
    print("   TABLE II: OVERALL PERFORMANCE OF THE PROPOSED METHOD   ")
    print("=======================================================")
    print(f"Number of test scenarios                   : {num_scenarios}")
    print(f"Success rate (%)                           : {success_rate:.2f}")
    print(f"Mean inference time (ms)                   : {mean_inference_time:.2f}")
    print(f"95th percentile inference time (ms)        : {p95_inference_time:.2f}")
    print(f"Median absolute passing angle |θ| (deg)    : {median_passing_angle:.1f}")
    print(f"95th percentile |θ| (deg)                  : {p95_passing_angle:.1f}")
    print(f"Median reward                              : {median_reward:.2f}")
    print(f"Reward interquartile range (P75-P25)       : {iqr_reward:.2f}")
    print("-------------------------------------------------------")
    print(">>  Additional (New) Metrics for Reference  <<")
    print(f"Mean safety margin (m)                     : {mean_margin:.3f}")
    print(f"Absolute Minimum margin encountered (m)    : {min_margin_overall:.3f}")
    print("=======================================================\n")
    
if __name__ == "__main__":
    run_evaluation(num_scenarios=300)
