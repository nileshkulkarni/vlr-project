import argparse

def get_opts():
  parser = argparse.ArgumentParser('Weakly supv AC')
  parser.add_argument('--cache_dir', type=str, default='cachedir')
  parser.add_argument('--ucf_dir', type=str, default='/scratch/nileshk/vlr-project/thumos_data/')
  parser.add_argument('--labels_dir', type=str, default='/scratch/nileshk/vlr-project/thumos_data/labels/')
  parser.add_argument('--feature_size', type=int, default=1024)
  parser.add_argument('--num_classes', type=int, default=20)
  parser.add_argument('--combine_strategy', type=str, default='uniform')
  parser.add_argument('--segments', type=int, default=400)
  parser.add_argument('--batch_size', type=int, default=4)
  parser.add_argument('--epochs', type=int, default=20)
  parser.add_argument('--log_every', type=int, default=20)
  parser.add_argument('--lr', type=float, default=0.01)
  parser.add_argument('--sparsity_wt_rgb', type=float, default=0.001)
  parser.add_argument('--sparsity_wt_flow', type=float, default=0.)
  parser.add_argument('--cls_wt_both', type=float, default=0)
  parser.add_argument('--cls_wt_flow', type=float, default=0)
  parser.add_argument('--cls_wt_rgb', type=float, default=1.0)
  parser.add_argument('--momentum', type=float, default=0.9)
  opts = parser.parse_args()

  return opts