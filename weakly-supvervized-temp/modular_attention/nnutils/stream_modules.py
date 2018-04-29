import torch
import torch.nn as nn
import pdb


class ActionClassification(nn.Module):
  def __init__(self, feature_size, num_classes, opts):
    super(ActionClassification, self).__init__()
    self.opts = opts
    self.attention_flow_stream = StreamModule(feature_size, num_classes)
    self.attention_rgb_stream = StreamModule(feature_size, num_classes)
    self.classifier_flow_stream = StreamClassificationHead(feature_size,
                                                           num_classes)
    self.classifier_rgb_stream = StreamClassificationHead(feature_size,
                                                          num_classes)
    self.classifier_both = StreamClassificationHead(feature_size, num_classes)
    self.multi_label_cross = nn.MultiLabelSoftMarginLoss()
  
  def forward(self, rgb_features, flow_features):
    # rgb B x T x 1024
    # flow B x T x 1024
    wt_ft_rgb, attn_rgb = self.attention_rgb_stream(rgb_features)
    wt_ft_flow, attn_flow = self.attention_flow_stream(flow_features)
    class_flow = self.classifier_flow_stream(wt_ft_flow)
    class_rgb = self.classifier_rgb_stream(wt_ft_rgb)
    features_both = wt_ft_flow + wt_ft_rgb
    class_both = self.classifier_both(features_both)
    if self.opts.log_now:
      self.opts.logger.histo_summary('rgb_wt_fetaures',
                                     wt_ft_rgb.data.cpu().numpy(),
                                     self.opts.iteration)
      
    outputs = dict()
    outputs['class_both'] = class_both
    outputs['class_rgb'] = class_rgb
    outputs['class_flow'] = class_flow
    outputs['attn_rgb'] = attn_rgb
    outputs['attn_flow'] = attn_flow
    
    return outputs
  
  def get_attention_models(self):
    models = {}
    models[
      'rgb_attention_model'] = self.attention_rgb_stream.class_attention_models
    models['flow_attention_model'] = self.attention_flow_stream.class_attention_models
    return models
  
  def build_loss(self, outputs, target_class):
    # target_class is B x C
    self.rgb_loss = self.multi_label_cross(outputs['class_rgb'], target_class)
    self.flow_loss = self.multi_label_cross(outputs['class_flow'], target_class)
    self.both_loss = self.multi_label_cross(outputs['class_both'], target_class)
    self.rgb_sparsity = self.attention_rgb_stream.attention_module.l1_sparsity_loss(
      outputs['attn_rgb'])
    self.flow_sparsity = self.attention_flow_stream.attention_module.l1_sparsity_loss(
      outputs['attn_flow'])
    losses = {}
    losses['rgb_loss'] = self.opts.cls_wt_rgb * self.rgb_loss
    losses['flow_loss'] = self.opts.cls_wt_flow * self.flow_loss
    losses['both_loss'] = self.opts.cls_wt_both * self.both_loss
    losses['rgb_sparsity_loss'] = self.opts.sparsity_wt_rgb * self.rgb_sparsity
    losses[
      'flow_sparsity_loss'] = self.opts.sparsity_wt_flow * self.flow_sparsity
    
    self.total_loss = losses['rgb_loss'] + losses['flow_loss'] + losses[
      'both_loss'] + losses['rgb_sparsity_loss'] + losses['flow_sparsity_loss']
    losses['total_loss'] = self.total_loss
    return losses


class StreamModule(nn.Module):
  def __init__(self, feature_size, num_classes):
    super(StreamModule, self).__init__()
    self.feature_size = feature_size
    self.num_classes = num_classes
    self.attention_module = ClassAttentionModule(self.feature_size,
                                                 self.num_classes)
    self.class_attention_models = ClassAttentionModule(self.feature_size,
                                                       self.num_classes)
  
  def forward(self, x):
    # x is B x T x feature_size
  
    attention = self.class_attention_models.forward(x)  # B x T x num_classes
    attention_expand = attention.unsqueeze(3).expand(
      torch.Size([attention.size(0),attention.size(1),attention.size(2), x.size(2)]))  # B x T x num_clas x feature_size
    x_expand = x.unsqueeze(2).expand(attention_expand.size())  # B x T x num_class x feature_size
    new_features = x_expand * attention_expand # B x T x num_classes x feature_size
    weighted_features = torch.sum(new_features, 1)
    return weighted_features, attention  # ( B x num_class x feature_size , B x T x num_class)

class ClassifierModule(nn.Module):
  # performs 0/1 classification
  def __init__(self, class_index, feature_size):
    super(ClassifierModule, self).__init__()
    self.mlp1 = nn.Linear(feature_size, 256)
    self.mlp2 = nn.Linear(256, 1)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.mlp1(x)
    x = self.relu(x)
    x = self.mlp2(x)
    return x

class StreamClassificationHead(nn.Module):
  def __init__(self, feature_size, num_classes):
    super(StreamClassificationHead, self).__init__()
    self.num_classes = num_classes
    self.classifier_modules = nn.ModuleList(
      [ClassifierModule(i, feature_size) for i in range(num_classes)])
  
  def forward(self, x):
    # x :  B x num_class x feature_size
    # x = self.classifier(x)   # B x num_class x 1
    outs = []
   
    for i in range(self.num_classes):
      outs.append(self.classifier_modules[i](x[:, i, :]))
    outs = torch.cat(outs, dim=1)  # B x num_classes x 1
    # B x num_class
    return outs

class ClassAttentionModule(nn.Module):
  def __init__(self, feature_size, num_classes):
    super(ClassAttentionModule, self).__init__()
    self.num_classes = num_classes
    self.attention_modules = nn.ModuleList([AttentionModule(i, feature_size) for i in range(num_classes)])

  def forward(self, x, class_index=None):
    if class_index is None:
      out = []
      for m in self.attention_modules:
        out.append(m(x))  ## m(x) B x T x 1024
      out = torch.stack(out).squeeze(3)
      out = out.permute(1, 2, 0)
      return out
    else:
      return self.attention_modules[class_index](x)

  def l1_sparsity_loss(self, x):
    return x.sum(dim=2).sum(dim=1)
  
  def build_binary_loss(self, class_index, pred_labels, target_labels, label_weights):
    return self.attention_modules[class_index].build_binary_loss(pred_labels, target_labels, label_weights)


class AttentionModule(nn.Module):
  def __init__(self, class_id, feature_size):
    super(AttentionModule, self).__init__()
    self.class_id = class_id
    self.feature_size = feature_size
    self.net = nn.Sequential(nn.Linear(self.feature_size, 256),
                             nn.ReLU(),
                             nn.Linear(256, 1),
                             nn.Sigmoid())  
  def l1_penalty(self, var):
    return torch.abs(var).sum()
  def forward(self, feature_segments):
    x = self.net(feature_segments)
    return x  ## B x T
  def build_binary_loss(self, pred_labels, target_labels, label_weights, lambda1 = 1.3e-3):
    ## pred_labels B x 1 ## target_labels B x 1
    return torch.nn.functional.binary_cross_entropy(pred_labels, target_labels, weight=label_weights) + lambda1 * self.l1_penalty(pred_labels)
