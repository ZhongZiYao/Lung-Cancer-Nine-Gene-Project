import os
import time
import torch
import torchvision
import numpy as np
import pandas as pd
from os.path import join as j_
from PIL import Image

# UNI repo imports
from uni import get_encoder
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader
from uni.downstream.eval_patch_features.linear_probe import eval_linear_probe
from uni.downstream.eval_patch_features.fewshot import eval_knn, eval_fewshot
from uni.downstream.eval_patch_features.protonet import ProtoNet
from uni.downstream.eval_patch_features.metrics import print_metrics, get_eval_metrics
from uni.downstream.utils import concat_images


def print_model_info(model):
	print("=" * 50)
	print("UNI 模型配置:")
	print("=" * 50)
	print(f"模型类型: {type(model).__name__}")
	print(f"特征维度: {getattr(model, 'embed_dim', 'N/A')}")
	if hasattr(model, 'patch_embed'):
		print(f"Patch size: {getattr(model.patch_embed, 'patch_size', 'N/A')}")
		print(f"图像尺寸: {getattr(model.patch_embed, 'img_size', 'N/A')}")
	print(f"Transformer 层数: {len(model.blocks) if hasattr(model, 'blocks') else 'N/A'}")
	if hasattr(model, 'blocks') and len(model.blocks) > 0 and hasattr(model.blocks[0], 'attn'):
		print(f"注意力头数: {getattr(model.blocks[0].attn, 'num_heads', 'N/A')}")

	print("\n" + "=" * 50)
	print("模型参数统计:")
	print("=" * 50)
	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"总参数量: {total_params:,}")
	print(f"可训练参数: {trainable_params:,}")
	print(f"冻结参数: {total_params - trainable_params:,}")

	print("=" * 50)
	print("UNI 模型层级结构:")
	print("=" * 50)
	for name, module in model.named_children():
		print(f"\n{name}: {type(module).__name__}")
		if name == 'blocks':
			print(f"  └─ Transformer Blocks 数量: {len(module)}")
			if len(module) > 0:
				print(f"     示例 Block 0 结构:")
				for sub_name, sub_module in list(module[0].named_children()):
					print(f"       └─ {sub_name}: {type(sub_module).__name__}")


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Device:", device)

	# Load UNI encoder and transform
	model, transform = get_encoder(enc_name='uni', device=device)
	print_model_info(model)

	# Data roots
	dataroot = '../assets/data/CRC100K/'
	assert os.path.isdir(j_(dataroot, 'NCT-CRC-HE-100K-NONORM')), "Train folder missing"
	assert os.path.isdir(j_(dataroot, 'CRC-VAL-HE-7K')), "Test folder missing"

	# Datasets and loaders
	train_dataset = torchvision.datasets.ImageFolder(j_(dataroot, 'NCT-CRC-HE-100K-NONORM'), transform=transform)
	test_dataset = torchvision.datasets.ImageFolder(j_(dataroot, 'CRC-VAL-HE-7K'), transform=transform)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=16)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=16)

	# Feature extraction
	start = time.time()
	train_features = extract_patch_features_from_dataloader(model, train_loader)
	test_features = extract_patch_features_from_dataloader(model, test_loader)
	train_feats = torch.tensor(train_features['embeddings'])
	train_labels = torch.tensor(train_features['labels']).long()
	test_feats = torch.tensor(test_features['embeddings'])
	test_labels = torch.tensor(test_features['labels']).long()
	elapsed = time.time() - start
	print(f"Feature extraction took {elapsed:.3f} seconds")

	# Linear probe
	linprobe_eval_metrics, _ = eval_linear_probe(
		train_feats=train_feats,
		train_labels=train_labels,
		valid_feats=None,
		valid_labels=None,
		test_feats=test_feats,
		test_labels=test_labels,
		max_iter=1000,
		verbose=True,
	)
	print("\n[Linear Probe Metrics]")
	print_metrics(linprobe_eval_metrics)

	# KNN & ProtoNet
	knn_eval_metrics, knn_dump, proto_eval_metrics, proto_dump = eval_knn(
		train_feats=train_feats,
		train_labels=train_labels,
		test_feats=test_feats,
		test_labels=test_labels,
		center_feats=True,
		normalize_feats=True,
		n_neighbors=20,
	)
	print("\n[KNN Metrics]")
	print_metrics(knn_eval_metrics)
	print("\n[ProtoNet Metrics]")
	print_metrics(proto_eval_metrics)

	# Few-shot episodes
	fewshot_episodes, fewshot_dump = eval_fewshot(
		train_feats=train_feats,
		train_labels=train_labels,
		test_feats=test_feats,
		test_labels=test_labels,
		n_iter=100,
		n_way=9,
		n_shot=16,
		n_query=test_feats.shape[0],
		center_feats=True,
		normalize_feats=True,
		average_feats=True,
	)
	print("\n[Few-shot Summary]")
	print(fewshot_dump)

	# ProtoNet fit and retrieval visualization (paths only; image grid requires a notebook display)
	proto_clf = ProtoNet(metric='L2', center_feats=True, normalize_feats=True)
	proto_clf.fit(train_feats, train_labels)
	print('Prototype shape:', proto_clf.prototype_embeddings.shape)

	# Example: show top-5 indices for one class if desired
	try:
		dist, topk_inds = proto_clf._get_topk_queries_inds(test_feats, topk=5)
		print('Top-5 indices per class (first class example):', topk_inds[0])
	except AssertionError as e:
		print('FAISS not available for top-k retrieval:', e)


if __name__ == "__main__":
	main()

