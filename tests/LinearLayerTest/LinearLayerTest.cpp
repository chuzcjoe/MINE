#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "Linear.h"
#include "Mat.h"
#include "VulkanBuffer.h"
#include "VulkanCommandBuffer.h"
#include "VulkanContext.h"
#include "VulkanImage.h"
#include "VulkanQueryPool.h"
#include "VulkanSync.h"
#include "VulkanUtils.h"

namespace mine {
namespace test {

void ReadFcnWeights(const std::string& path, std::vector<float>& data) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open weights file: " + path);
    }

    file.seekg(0, std::ios::end);
    const auto byte_count = file.tellg();

    file.seekg(0, std::ios::beg);
    const auto element_count = byte_count / sizeof(float);
    data.resize(element_count);
    if (!file.read(reinterpret_cast<char*>(data.data()), byte_count)) {
        throw std::runtime_error("Failed to read weights file: " + path);
    }
}

// TODO: hardcode for now, later we can parse from fcn.json

// torch.Size([60, 10])
// torch.Size([60])
// torch.Size([2, 60])
// torch.Size([2])

// for fc layers, PyTorch uses (out_features, in_features) for weight shape
struct LinearLayerGraph {
    int fc1_in_features = 10;
    int fc1_out_features = 60;
    int fc1_bias = 60;
    int fc2_in_features = 60;
    int fc2_out_features = 2;
    int fc2_bias = 2;
};

TEST(LinearLayerTest, test) {
    std::vector<float> weights;
    ReadFcnWeights("tests/LinearLayerTest/fcn_weights.bin", weights);
    printf("fcn weights size: %lu\n", weights.size());
    printf("first three weights: %f, %f, %f\n", weights[0], weights[1], weights[2]);

    // Setup Vulkan
    LinearLayerGraph graph;
  core::vulkan::QueueFamilyType queue_family_type = core::vulkan::QueueFamilyType::Compute;
  core::vulkan::VulkanContext context(true, queue_family_type, nullptr);
  context.Init();
  core::vulkan::VulkanCommandBuffer command_buffer(&context);
  core::vulkan::VulkanFence fence(&context);
  core::vulkan::VulkanQueryPool query_pool(&context, VK_QUERY_TYPE_TIMESTAMP);
  
  const VkDeviceSize src_buffer_size = graph.fc1_in_features * sizeof(float);
  const VkDeviceSize weights_buffer_size = graph.fc1_in_features * graph.fc1_out_features * sizeof(float);
  const VkDeviceSize bias_buffer_size = graph.fc1_bias * sizeof(float);
  const VkDeviceSize dst_buffer_size = graph.fc1_out_features * sizeof(float);

  // Create buffers
  core::vulkan::VulkanBuffer src_buffer(
      &context, src_buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  core::vulkan::VulkanBuffer weights_buffer(
      &context, weights_buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  core::vulkan::VulkanBuffer bias_buffer(
      &context, bias_buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  core::vulkan::VulkanBuffer dst_buffer(
      &context, dst_buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Fill buffers
    src_buffer.MapData([](void* data) { 
        core::Mat<float, 1> mat(1, 10);
        mat.Fill(1.0f);
        memcpy(data, mat.data(), mat.total() * sizeof(float)); 
    });
    weights_buffer.MapData([&weights, &graph](void* data) {
        memcpy(data, weights.data(), graph.fc1_in_features * graph.fc1_out_features * sizeof(float));
    });
    bias_buffer.MapData([&weights, &graph](void* data) {
        memcpy(data, weights.data() + graph.fc1_in_features * graph.fc1_out_features, graph.fc1_bias * sizeof(float));
    });
    dst_buffer.MapData([&graph](void* data) {
        memset(data, 0, graph.fc1_out_features * sizeof(float));
    });

  // Create and run compute sum pipeline
  std::unique_ptr<mine::vulkan::Linear> linear = std::make_unique<mine::vulkan::Linear>(&context, 
                                                                                      src_buffer,
                                                                                      weights_buffer,
                                                                                      bias_buffer,
                                                                                      dst_buffer,
                                                                                      graph.fc1_in_features,
                                                                                      graph.fc1_out_features);
  linear->Init();

  fence.Reset();

  vkResetCommandBuffer(command_buffer.buffer(), 0);
  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  vkBeginCommandBuffer(command_buffer.buffer(), &begin_info);
  query_pool.Reset(command_buffer.buffer());
  query_pool.Query(command_buffer.buffer(), 0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  linear->Run(command_buffer.buffer());
  query_pool.Query(command_buffer.buffer(), 1, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  VkSubmitInfo submit_info{};
  command_buffer.Submit(fence.fence, submit_info);

  vkWaitForFences(context.logical_device, 1, &fence.fence, VK_TRUE, UINT64_MAX);

  // Check data
  core::Mat<float, 1> output(1, graph.fc1_out_features);
  dst_buffer.MapData([&output, &graph](void* data) {
      memcpy(output.data(), data, graph.fc1_out_features * sizeof(float));
  });
  printf("GPU Result: %f\n", *output(0, 0));

  query_pool.GetQueryResults();
  const auto timestamps = query_pool.GetTimeStamps();
  const auto runtime_ms = (timestamps[1] - timestamps[0]) * (context.timestamp_period / 1000000.0);
  printf("GPU time: %fms\n", runtime_ms);
}

}  // namespace test
}  // namespace mine
