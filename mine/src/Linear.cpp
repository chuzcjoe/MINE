#include "Linear.h"

namespace mine {
namespace vulkan {

Linear::Linear(core::vulkan::VulkanContext* context, 
             core::vulkan::VulkanBuffer& src,
             core::vulkan::VulkanBuffer& weights,
             core::vulkan::VulkanBuffer& bias,
             core::vulkan::VulkanBuffer& dst,
             const int in_features,
             const int out_features)
    : VulkanCompute(context),
      src_buffer(src),
      weights_buffer(weights),
      bias_buffer(bias),
      dst_buffer(dst),
      uniform_buffer_(context, sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
      uniform_data_{.in_features = in_features, .out_features = out_features} {}

void Linear::Init() {
  VulkanCompute::Init();

  CreateUniformBufferDescriptorSet(0, uniform_buffer_);
  CreateStorageBufferDescriptorSet(1, src_buffer);
  CreateStorageBufferDescriptorSet(2, weights_buffer);
  CreateStorageBufferDescriptorSet(3, bias_buffer);
  CreateStorageBufferDescriptorSet(4, dst_buffer);

  vkUpdateDescriptorSets(context_->logical_device, writes_.size(), writes_.data(), 0, nullptr);
}

void Linear::Run(const VkCommandBuffer command_buffer) {
  // Copy uniform data to the uniform buffer
  uniform_buffer_.MapData(
      [this](void* data) { memcpy(data, &uniform_data_, sizeof(UniformData)); });

  // Record commands to dispatch the compute shader
  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                          &descriptor_set_, 0, nullptr);

  if (uniform_data_.in_features <= 0 || uniform_data_.out_features <= 0) {
    throw std::runtime_error("Invalid features in Linear layer");
  }

  const uint32_t group_x = 1;
  const uint32_t group_y = static_cast<uint32_t>(uniform_data_.out_features);
  vkCmdDispatch(command_buffer, group_x, group_y, 1);
}

std::vector<core::vulkan::BindingInfo> Linear::GetBindingInfo() const {
  return {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
}

const std::vector<uint32_t>& Linear::LoadShaderCode() const {
  // Load and return the SPIR-V code for the compute shader
  // This is a placeholder; actual implementation will read from a file or embedded resource
  static const std::vector<uint32_t> shader_code =
#include "Linear.comp.spv"
      ;
  return shader_code;
}

}  // namespace vulkan
}  // namespace mine
