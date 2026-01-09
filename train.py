from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from env_passenv_grid import PassEnvGrid


if __name__ == "__main__":
    vec_env = make_vec_env(
        lambda: PassEnvGrid(grid_n=16, grid_samples=3),
        n_envs=4,
        seed=0
    )

    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log="tb_sac_grid",
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
    )

    save_name = "pass_planner_sac_grid"

    try:
        model.learn(total_timesteps=400000)
        print("SAC 训练完成！")
    except KeyboardInterrupt:
        print("\n[INFO] 训练被 Ctrl+C 中断，准备保存当前模型...")
    finally:
        # ✅ 无论正常结束还是中断，都保存
        try:
            model.save(save_name)
            print(f"[INFO] 已保存模型：{save_name}.zip")
        except Exception as e:
            print(f"[WARN] 保存模型失败：{e}")

        try:
            vec_env.close()
        except Exception as e:
            print(f"[WARN] 关闭环境失败：{e}")
