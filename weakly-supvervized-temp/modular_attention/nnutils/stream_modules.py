import torch
import torch.nn as nn


class ActionClassification(nn.Module):
  def __init__(self, feature_size, num_classes, opts):
    super(ActionClassification, self).__init__()
    self.opts = opts
    self.attention_flow_stream = StreamModule(feature_size)
    self.attentition_rgb_stream = StreamModule(feature_size)
    self.classifier_flow_stream = StreamClassificationHead(feature_size,
                                                           num_classes)
    self.classifier_rgb_stream = StreamClassificationHead(feature_size,
                                                          num_classes)
    self.classifier_both = StreamClassificationHead(feature_size, num_classes)
    self.multi_label_cross = nn.MultiLabelSoftMarginLoss()
  
  def forward(self, rgb_features, flow_features):
    # rgb B x T x 1024
    # flow B x T x 1024
    
    wt_ft_rgb, attn_rgb = self.attentition_rgb_stream(rgb_features)
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
  
  def build_loss(self, outputs, target_class):
    # target_class is B x C
    self.rgb_loss = self.multi_label_cross(outputs['class_rgb'], target_class)
    self.flow_loss = self.multi_label_cross(outputs['class_flow'], target_class)
    self.both_loss = self.multi_label_cross(outputs['class_both'], target_class)
    self.rgb_sparsity = self.attentition_rgb_stream.attention_module.l1_sparsity_loss(
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
  def __init__(self, feature_size):
    super(StreamModule, self).__init__()
    self.feature_size = feature_size
    self.attention_module = AttentionModule(self.feature_size)
  
  def forward(self, x):
    # x is B x T x 1024
    attention = self.attention_module(x)  # B x T
    attention_expand = attention.expand(x.size())  # B X T x feature_size
    new_features = x * attention_expand
    weighted_features = torch.sum(new_features, 1)
    return weighted_features, attention


class StreamClassificationHead(nn.Module):
  def __init__(self, feature_size, num_classes):
    super(StreamClassificationHead, self).__init__()
    self.classifier = nn.Linear(feature_size, num_classes)
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, x):
    x = self.classifier(x)
    x = self.sigmoid(x)
    return x


class ClassAttentionModule(nn.Module):
  def __init__(self, feature_size, num_classes):
    super(ClassAttentionModule, self).__init__()
    self.num_classes = num_classes
    self.modules = nn.ModuleList(
      [nn.Linear(100, feature_size) for i in range(self.num_classes)])
    
  def forward(self,class_index, x):
    import pdb;pdb.set_trace()
    modules = [m for m in enumerate(self.modules)]
    return modules[class_index](x)
  
  
  def build_binary_loss(self, class_index, pred_labels, target_labels):
    return self.modules[class_index].build_binary_loss(pred_labels, target_labels)

class AttentionModule(nn.Module):
  def __init__(self, class_id, feature_size):
    super(AttentionModule, self).__init__()
    self.class_id = class_id
    self.feature_size = feature_size
    #self.fc1 = nn.Linear(self.feature_size, 256)
    #self.relu = nn.ReLU()
    #self.fc2 = nn.Linear(256, 1)
    #self.sigmoid = nn.Sigmoid()
    self.net = nn.Sequential(nn.Linear(self.feature_size, 256),nn.ReLU(),nn.Linear(256, 1),nn.Sigmoid())
  
  def forward(self, feature_segments):
    ## B x T x 1024
    #x = self.fc1(feature_segments)
    #x = self.relu(x)
    #x = self.fc2(x)
    #x = self.sigmoid(x)
    x = self.net(feature_segments)
    return x  ## B x T

  def build_binary_loss(self, pred_labels, target_labels):
    ## pred_labels B x 1 ## target_labels B x 1
    return torch.nn.functional.binary_cross_entropy(pred_labels, target_labels)