#pragma once

#include <iostream>

#include "VulkanBuffer.h"
#include "VulkanCompute.h"

namespace mine {
namespace vulkan {

class Linear : public core::vulkan::VulkanCompute {
 public:
  Linear(core::vulkan::VulkanContext* context, 
             core::vulkan::VulkanBuffer& src,
             core::vulkan::VulkanBuffer& weights,
             core::vulkan::VulkanBuffer& bias,
             core::vulkan::VulkanBuffer& dst,
             const int in_features,
             const int out_features);

  void Init() override;
  void Run(const VkCommandBuffer command_buffer);

  core::vulkan::VulkanBuffer& src_buffer;
  core::vulkan::VulkanBuffer& weights_buffer;
  core::vulkan::VulkanBuffer& bias_buffer;
  core::vulkan::VulkanBuffer& dst_buffer;

 protected:
  std::vector<core::vulkan::BindingInfo> GetBindingInfo() const override;
  const std::vector<uint32_t>& LoadShaderCode() const override;

 private:
  core::vulkan::VulkanBuffer uniform_buffer_;
  struct UniformData {
    int in_features;
    int out_features;
  } uniform_data_;
};
}  // namespace vulkan
}  // namespace mine