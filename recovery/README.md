# 从 GitHub 恢复 Canopy 五机集群

从 [恢复 Release](https://github.com/fiberpine/Canopy/releases/tag/cluster-recovery-20261118) 获取资产。本分支包含连续双动作 v4 源码、训练与评测入口、健康基座 seed3 step-6013196 的训练和推理权重、云端冻结准入证据，以及可用于历史诊断的联合训练 checkpoint。

## 在新机器上恢复

1. 新建五台 Ubuntu 18.04 x86_64 VM。选用允许自定义内核的实例，准备串口或控制台启动入口。为新机器分配 SSH 凭据，从云控制台核对新主机指纹。给训练日志和 checkpoint 留磁盘空间。
2. 五台以 `ubuntu` 用户检出同一 Git tag 到同一绝对目录。这样 learner 启动远端 actor 时，程序与解释器的绝对路径一致。

   ```bash
   git clone --depth 1 --branch cluster-recovery-20261118 https://github.com/fiberpine/Canopy.git /home/ubuntu/canopy
   cd /home/ubuntu/canopy
   bash recovery/restore.sh --system
   bash recovery/restore.sh --kernel
   ```

3. 在新机器控制台引导进入 `4.13.1-0521-learner`。仓库资产保存了原五台共同使用的内核与模块，新机器本地生成 initramfs。结束后在每台运行：

   ```bash
   cd /home/ubuntu/canopy
   uname -r
   bash recovery/restore.sh --workspace
   ```

   脚本校验 Release SHA256，恢复 Python3.6/TF1.14，校验 native 的构建输入和保护基座文件，加载模型做初始推理，并验证 v4 的准入契约。如果启动设置不合格，修正后执行 `bash recovery/restore.sh --verify`。

4. 在 learner 配置新五机私网路由。复制 `recovery/cluster.example.json` 到仓库外，分别填写新私网地址、私钥和已核验的 known_hosts 路径。确认 learner 可通过 SSH 连到所有 actor，actor 可经私网访问 learner 的 TCP 25000。训练控制器按条件轮转分配四台 actor，每台六个环境。端口按配置与 actor 索引分配，保持实验流量在新集群内。
5. 先跑短机制和容量检查。获批的训练窗口须在私有配置中明确登记 `allow_tcp=true`、`purpose`、`start_epoch`、`deadline_epoch`。准入回执随源码发布，但训练授权由当次实验单独签发。入口如下：

   ```bash
   runtime/venv/bin/python -B joint_healthy.py train \
      --profile continuous-v4 --run-id NEW-UNIQUE-ID \
      --seconds-per-actor 600 --authorization /path/to/new-window.json \
      --release-joint --gate-receipt recovery/evidence/initialization-admission.json
   ```

   这会从健康窗口初始化新训练。原初始冻结测试为 24 条件×1，完整结果见 [REPORT.md](evidence/REPORT.md)。历史 v4 同轮未训练新模型，选定旧模型用于诊断，训练前无需复制旧 replay。

## 资产及可复现范围

- `runtime/venv`、定制内核和 Mahimahi 二进制从 Release 下载，仓库内包清单提供 SHA256。主机 Python3.6 与系统共享库由 Ubuntu 18.04 包提供。
- 内核原始源码为 Linux 4.13.1。编译使用 [训练内核配置](provenance/kernel-4.13.1-0521-learner.config) 和 [Orca 补丁](kernel-source/orca-0521.patch)。原始 tarball 的 SHA256 为 `decee7a2de34aea921fce3e7934f520790fe43176de29bd9718a84419ca6e1ce`，上游地址为 [kernel.org](https://cdn.kernel.org/pub/linux/kernel/v4.x/linux-4.13.1.tar.xz)。Mahimahi 对应 [287ef99](https://github.com/ravinet/mahimahi/tree/287ef9916ffdef7f1dd75bc48e2ae0108b542a82)。
- `models/healthy-baseline-seed3` 含健康基座。 `recovery/models` 含选定历史联合模型和去除旧主机信息的 manifest。历史模型须用来源分支和配置加载，详见索引。TCP 状态、旧进程和逐包历史不能续接。
- 新实例的固件、VirtIO 设备、磁盘启动模式和供应商选项仍要现场核对。保留供应商内核作为回退。实际重启新 VM 是恢复核查的最后一步。
