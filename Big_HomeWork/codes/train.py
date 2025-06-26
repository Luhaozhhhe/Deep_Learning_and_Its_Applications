import torch
import time
import logging
import torch.nn.utils # 确保在文件顶部导入

class Trainer:
    def __init__(self, model, criterion, optimizer, device, logger=None, scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler  # 添加调度器
        self.logger = logger or logging.getLogger(__name__)
        self.model.to(device)

    def train(self, train_loader, valid_loader, num_epochs):
        train_loss_list = []
        train_top1_accuracy_list = []
        train_top5_accuracy_list = []
        valid_loss_list = []
        valid_top1_accuracy_list = []
        valid_top5_accuracy_list = []
        
        self.logger.info("Starting training...")
        for epoch in range(num_epochs):
            start_time = time.time()

            self.model.train()
            train_loss = 0.0
            train_top1_correct = 0
            train_top5_correct = 0
            total = 0
            
            for i, (images, labels) in enumerate(train_loader):
                batch_size = labels.size(0)
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                if i % 10 == 0:
                    if torch.isnan(images).any():
                        self.logger.debug(f'Batch {i}: input images contain NaN!')
                    else:
                        self.logger.debug(f'Batch {i}: input image stats - min: {images.min().item():.4f}, '
                                f'max: {images.max().item():.4f}, mean: {images.mean().item():.4f}')

                self.optimizer.zero_grad()
                outputs = self.model(images)

                # --- 在这里添加输出值检查 ---
                if i % 10 == 0: # 每 1 个批次打印一次，避免输出过多
                    self.logger.debug(f'Batch {i}: outputs min: {outputs.min().item():.4f}, '
                          f'max: {outputs.max().item():.4f}, mean: {outputs.mean().item():.4f}')
                # ---------------------------

                loss = self.criterion(outputs, labels)

                # --- 在这里添加损失值检查 ---
                if i % 10 == 0: # 每 10 个批次打印一次
                    self.logger.debug(f'Batch {i}: loss before backward: {loss.item():.4f}')
                # ---------------------------



                loss.backward()

                # --- 在这里添加梯度检查 ---
                 # 检查一些参数的梯度范数，例如 head 层的梯度
                if i % 10 == 0:
                    if hasattr(self.model, 'head') and self.model.head.weight.grad is not None:
                        grad_norm_head = self.model.head.weight.grad.norm().item()
                        self.logger.debug(f'Batch {i}: head.weight grad norm: {grad_norm_head:.4f}')
                # 您也可以遍历所有参数并检查它们的梯度
                    total_grad_norm = 0.0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            total_grad_norm += param.grad.norm().item() ** 2
                    total_grad_norm = total_grad_norm ** 0.5
                    self.logger.debug(f'Batch {i}: total grad norm: {total_grad_norm:.4f}')
                # --------------------------

                # --- 添加以下一行进行梯度裁剪 ---
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # max_norm=1.0 是一个常用的起始值
                # --------------------------------

                self.optimizer.step()

                train_loss += loss.item() * batch_size
                total += batch_size

                # Top-1 accuracy
                _, predicted = outputs.max(1)
                train_top1_correct += predicted.eq(labels).sum().item()

                # Top-5 accuracy
                _, top5_pred = outputs.topk(5, 1, largest=True, sorted=True)
                labels_expanded = labels.view(labels.size(0), -1).expand_as(top5_pred)
                correct = top5_pred.eq(labels_expanded).float()
                train_top5_correct += correct.sum().item()

                if i % 100 == 0:
                    self.logger.debug(f'Epoch [{epoch + 1}/{num_epochs}], '
                                    f'Step [{i}/{len(train_loader)}], '
                                    f'Batch Loss: {loss.item():.4f}')

            train_loss /= total
            train_top1_acc = 100. * train_top1_correct / total
            train_top5_acc = 100. * train_top5_correct / total

            valid_loss, valid_top1_acc, valid_top5_acc = self.evaluate(valid_loader, verbose=False)

            train_loss_list.append(train_loss)
            train_top1_accuracy_list.append(train_top1_acc)
            train_top5_accuracy_list.append(train_top5_acc)
            valid_loss_list.append(valid_loss)
            valid_top1_accuracy_list.append(valid_top1_acc)
            valid_top5_accuracy_list.append(valid_top5_acc)

            end_time = time.time()
            elapsed_time = end_time - start_time

            log_message = (f'Epoch {epoch + 1}/{num_epochs}, '
                         f'Train Loss: {train_loss:.4f}, '
                         f'Train Top-1 Acc: {train_top1_acc:.2f}%, '
                         f'Train Top-5 Acc: {train_top5_acc:.2f}%, '
                         f'Valid Loss: {valid_loss:.4f}, '
                         f'Valid Top-1 Acc: {valid_top1_acc:.2f}%, '
                         f'Valid Top-5 Acc: {valid_top5_acc:.2f}%, '
                         f'Time: {elapsed_time:.2f}s')
            
            self.logger.info(log_message)
            # 如果有调度器，更新学习率
            if self.scheduler is not None:
                self.scheduler.step()

        self.logger.info('Training Finished')
       
        return (train_loss_list, train_top1_accuracy_list, train_top5_accuracy_list,
                valid_loss_list, valid_top1_accuracy_list, valid_top5_accuracy_list)

    def evaluate(self, test_loader, verbose=False):
        test_loss = 0.0
        top1_correct = 0
        top5_correct = 0
        total = 0
        
        self.model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                batch_size = labels.size(0)
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item() * batch_size
                total += batch_size

                # Top-1 accuracy
                _, predicted = outputs.max(1)
                top1_correct += predicted.eq(labels).sum().item()

                # Top-5 accuracy
                _, top5_pred = outputs.topk(5, 1, largest=True, sorted=True)
                labels = labels.view(labels.size(0), -1).expand_as(top5_pred)
                correct = top5_pred.eq(labels).float()
                top5_correct += correct.sum().item()

        test_loss /= total
        top1_accuracy = 100. * top1_correct / total
        top5_accuracy = 100. * top5_correct / total

        if verbose:
            self.logger.info(f'Test Loss: {test_loss:.4f}, '
                            f'Top-1 Accuracy: {top1_accuracy:.2f}%, '
                            f'Top-5 Accuracy: {top5_accuracy:.2f}%')

        return test_loss, top1_accuracy, top5_accuracy