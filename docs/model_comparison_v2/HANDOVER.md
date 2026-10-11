# 新对话交接

先读本目录RESULTS.md、HANDOFF.json和PLAN.md，核验registration、acceptance、NOMINATION及资源收据哈希。

本轮开发已完成，预算已用完，禁止重新启动训练/评分或用新目录重置。所有旧资产保护记录在protection_after.json；没有修改既有状态入口，本目录是本轮新增结果入口。

至多一个提名只是开发结果。2024复核需要新授权和适配单模型/融合身份的独立预登记；不打开第二名，不调参。不自动替换主备、重训全量、推断测试、生成submission、上传或提交Git。

最终资源见RESOURCE_RECEIPT.json；失败恢复版本链见EXECUTION_HISTORY.md，FINAL_DELIVERY_CHECKS.json是交付后完整哈希核验。临时artifact_index.json来自finalize工作进程，最终索引以HANDOFF定位的final_artifact_index.json为准。
