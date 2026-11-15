/** Requests a WebGPU device with the largest buffer limits the adapter allows. */
export async function requestDevice(gpu: GPU): Promise<{ device: GPUDevice; adapterInfo: string }> {
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("WebGPU アダプタが見つかりません");
  const l = adapter.limits;
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: l.maxStorageBufferBindingSize,
      maxBufferSize: l.maxBufferSize,
      maxStorageBuffersPerShaderStage: Math.min(l.maxStorageBuffersPerShaderStage, 8),
      maxComputeWorkgroupStorageSize: Math.min(l.maxComputeWorkgroupStorageSize, 16384),
    },
  });
  const info = adapter.info;
  const adapterInfo = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(" ");
  return { device, adapterInfo };
}
