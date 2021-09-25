import torch
import torch.nn as nn
import torch.optim as optim
from config import env_config, train_config
from torch.utils.data import DataLoader

from model.GRL import GRL


def train_epoch(source_loader: DataLoader, target_loader: DataLoader,
                extractor: nn.Module, predictor: nn.Module,
                classifier: nn.Module):
    extractor.train()
    predictor.train()
    classifier.train()

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    optimizer_F = optim.Adam(extractor.parameters())
    optimizer_C = optim.Adam(predictor.parameters())
    optimizer_D = optim.Adam(classifier.parameters())

    # D loss: Domain Classifier的loss
    # F loss: Feature Extrator & Label Predictor的loss
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0
    for i, ((source_data, source_label), (target_data, _)) in \
            enumerate(zip(source_loader, target_loader)):

        source_data = source_data.to(env_config.device)
        source_label = source_label.to(env_config.device)
        target_data = target_data.to(env_config.device)

        # Mixed the source data and target data, or it'll mislead the running params
        #   of batch_norm. (runnning mean/var of soucre and target data are different.)
        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]) \
                            .to(env_config.device)
        # set domain label of source data to be 1.
        domain_label[:source_data.shape[0]] = 1

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        # Step 1 : train domain classifier
        feature = extractor(mixed_data)
        # We don't need to train feature extractor in step 1.
        # Thus we detach the feature neuron to avoid backpropgation.
        domain_logits = classifier(feature.detach())
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss += loss.item()
        loss.backward()
        optimizer_D.step()

        # Step 2 : train feature extractor and label classifier
        class_logits = predictor(feature[:source_data.shape[0]])
        domain_logits = classifier(feature)
        # loss = cross entropy of classification - lamb * domain binary cross entropy.
        #  The reason why using subtraction is similar to generator loss in disciminator of GAN
        loss = class_criterion(class_logits, source_label) \
               - train_config.lamb * domain_criterion(domain_logits, domain_label)
        running_F_loss += loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        total_hit += torch.sum(
            torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        print(i, end='\r')  # new output will flush the old one

    return running_D_loss / (i + 1), running_F_loss / (i + 1), \
           total_hit / total_num


def train_epoch_GRL(source_loader: DataLoader, target_loader: DataLoader,
                    extractor: nn.Module, predictor: nn.Module,
                    classifier: nn.Module):
    extractor.train()
    predictor.train()
    classifier.train()

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    optimizer_F = optim.Adam(extractor.parameters())
    optimizer_C = optim.Adam(predictor.parameters())
    optimizer_D = optim.Adam(classifier.parameters())

    # D loss: Domain Classifier的loss
    # F loss: Feature Extrator & Label Predictor的loss
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0
    for i, ((source_data, source_label), (target_data, _)) in \
            enumerate(zip(source_loader, target_loader)):

        source_data = source_data.to(env_config.device)
        source_label = source_label.to(env_config.device)
        target_data = target_data.to(env_config.device)

        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]) \
                            .to(env_config.device)
        domain_label[:source_data.shape[0]] = 1

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        # Step 1 : train feature extractor and label classifier
        feature = extractor(mixed_data)
        class_logits = predictor(feature[:source_data.shape[0]])
        domain_logits = classifier(feature)
        loss = class_criterion(class_logits, source_label)
        running_F_loss += loss.item()
        loss.backward(retain_graph=True)

        # Step 2 : train domain classifier
        # classifier has to be trained later due to GRL
        feature = GRL.apply(feature)
        domain_logits = classifier(feature)
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss += loss.item()
        loss.backward()

        optimizer_F.step()
        optimizer_C.step()
        optimizer_D.step()

        total_hit += torch.sum(
            torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        print(i, end='\r')  # new output will flush the old one

    return running_D_loss / (i + 1), running_F_loss / (i + 1), \
           total_hit / total_num
