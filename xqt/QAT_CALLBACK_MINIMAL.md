# QAT 与蒸馏最小用例

本文只说明 XQT 和 XDL 之间最小的 QAT/蒸馏协作形态. 它不是完整训练教程,也不把训练循环放进 XQT.

## 边界

```text
XQT or quant backend
    -> prepare qat-ready model
XDL CoreModel + callback
    -> run forward/loss/backward/step
XQT
    -> convert/export/analyze/benchmark
```

XQT 不负责 `optimizer.zero_grad() -> backward() -> step()`. QAT 和蒸馏训练逻辑直接写在用户自己的 `CoreModel.training_step()` 里. XDL 只需要提供少量 callback/helper 和 loss.

## QAT 最小训练任务

```python
import torch
import torch.nn as nn

from xdl.trainer import CoreModel


class QATTask(CoreModel):
    def __init__(self, model: nn.Module, lr: float = 1e-4) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = float(lr)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def training_step(self, batch, batch_idx: int) -> None:
        inputs, targets = batch
        logits = self.forward(inputs)
        loss = self.loss_fn(logits, targets)
        self.manual_optimization_step(
            loss,
            optimizer=self.optimizers[0],
            model=self.model,
        )
        self.log("loss", loss, prefix="train")

    def validation_step(self, batch, batch_idx: int) -> None:
        inputs, targets = batch
        logits = self.forward(inputs)
        loss = self.loss_fn(logits, targets)
        self.log("loss", loss, prefix="val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
```

## 最小 callback

`QATLifecycleCallback` 在 XDL 中实现. callback 只做状态切换,不改 loss,不执行 optimizer step.

```python
from xdl.callbacks import QATLifecycleCallback
from xdl.trainer import Trainer


trainer = Trainer(
    max_epochs=5,
    callbacks=[
        QATLifecycleCallback(
            disable_observer_epoch=3,
            freeze_bn_epoch=2,
        )
    ],
)
trainer.fit(qat_task, train_loader, val_loader)
```

这里用 duck typing 兼容不同量化 backend. 只要模块暴露 `disable_observer()` 或 `freeze_bn_stats()` 这类方法,callback 就能工作.

## 蒸馏最小训练任务

蒸馏训练不需要 XQT 参与训练循环. 直接用 XDL 的 `CoreModel`,在 `training_step()` 里组合 hard label loss 和 KD loss.

```python
import torch
import torch.nn as nn

from xdl.loss import distillation_loss
from xdl.trainer import CoreModel


class DistillTask(CoreModel):
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        lr: float = 1e-4,
        temperature: float = 2.0,
        alpha: float = 0.7,
    ) -> None:
        super().__init__()
        self.student = student
        self.teacher = teacher.eval()
        self.lr = float(lr)
        self.temperature = float(temperature)
        self.alpha = float(alpha)

        for param in self.teacher.parameters():
            param.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.student(inputs)

    def training_step(self, batch, batch_idx: int) -> None:
        inputs, targets = batch
        student_logits = self.student(inputs)
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)

        loss = distillation_loss(
            student_logits,
            teacher_logits,
            targets=targets,
            temperature=self.temperature,
            alpha=self.alpha,
        ).total

        self.manual_optimization_step(
            loss,
            optimizer=self.optimizers[0],
            model=self.student,
        )
        self.log("loss", loss, prefix="train")

    def validation_step(self, batch, batch_idx: int) -> None:
        inputs, targets = batch
        logits = self.student(inputs)
        loss = nn.functional.cross_entropy(logits, targets)
        self.log("loss", loss, prefix="val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.student.parameters(), lr=self.lr)
```

这段代码是任务训练逻辑,应该留在 XDL 或用户项目里. XQT 只通过 training provider 触发它,或在训练完成后接收 student 产物继续导出和评估.

## XQT 侧只接产物

QAT 或蒸馏训练完成后,XQT 只继续处理模型产物:

```python
from xqt import XQTOptimizationSession

# 伪代码: 训练在 XDL 完成
trained_model = qat_task.model
# 或:
trained_model = distill_task.student

# XQT 后续负责转换,导出,误差分析和 benchmark.
# 真实 session 还需要 project,model_config,task,data_splits 等上下文.
session = XQTOptimizationSession(
    project={"name": "after_training", "artifact_dir": "artifacts/xqt/after_training"},
    model=trained_model,
    model_config={"dtype": "float32", "device": "cpu"},
    task={"type": "classification"},
)
session.export(...)
session.runtime_eval(...)
session.benchmark(...)
```

如果 XQT workflow 需要触发训练,它也只应该通过 training provider 委托给 XDL 或第三方训练框架,不要在 XQT 内部实现训练循环.

## Recovery

Recovery 也遵循同一原则:

- recovery 本质是短程 fine-tune,可用参数冻结 callback 或 helper 辅助.
- XQT 只收集 provider report,再继续做压缩产物转换,误差评估和导出.
