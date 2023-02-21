#include <future>
#include <initializer_list>
#include <type_traits>
#include <ctime>
#include <atomic>
#include <iostream>
#include <fstream>
#include <map>
#include <deque>

class TraceGPUMemoTool
{
public:
	TraceGPUMemoTool(int device_id);
	~TraceGPUMemoTool() = default;
 
public:
  void Record(size_t op_idx, const std::string& op_name);
  void Pause();
  void Stop();
  size_t GetGPUMemoryInfo();

private:
  std::condition_variable cv_;
  std::mutex mutex_;
  bool is_stop_{false};
  bool is_pause_{true};
  std::string op_name_;
  size_t op_idx_;
  std::map<size_t, std::deque<size_t>>memory_info_;
  std::map<size_t, std::string>op_idx_2_op_name_;
};

size_t TraceGPUMemoTool::GetGPUMemoryInfo() {
  size_t avail = 0;
  size_t total = 0;
  size_t used = 0;
  // cudaMemGetInfo(&avail, &total);
  // used = (total - avail) / 1024 / 1024;
  return used;
}

TraceGPUMemoTool::TraceGPUMemoTool(int device_id)
{
  // cudaSetDevice(device_id);
  std::thread t([&]() {
    while(true) {
      std::unique_lock<std::mutex> lock(mutex_);
      if(is_stop_) {
        lock.unlock();
        break;
      }
      // cv_.wait(lock, [this] {return !is_pause_; });
      while(is_pause_)
        cv_.wait(lock);
      memory_info_[op_idx_].push_back(GetGPUMemoryInfo());
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    });
  t.detach();
}

void TraceGPUMemoTool::Stop()
{
  std::unique_lock<std::mutex> lock(mutex_);
  is_stop_ = true;
  std::cout << "Stop\n";
  std::ofstream destFile("./out.txt", std::ios::out);
  std::cout << "memory_info_.size(): " << memory_info_.size() << "\n";
  for(auto& pair : memory_info_) {
    destFile << "{op_idx: " << pair.first << " ,op_name: " << op_idx_2_op_name_[pair.first] << "\n";
    for(auto i : pair.second) {
      destFile << i << " ";
    }
    destFile << "}\n";
  }
  
  memory_info_.clear();
  std::cout << "Stop end\n";
}
void TraceGPUMemoTool::Pause()
{
    std::unique_lock<std::mutex> lock(mutex_);
    std::cout << "Pause\n";
    is_pause_ = true;
    cv_.notify_one();
    std::cout << "Pause end\n";
}
void TraceGPUMemoTool::Record(size_t op_idx, const std::string& op_name)
{
    std::cout << "try\n";
    std::unique_lock<std::mutex> lock(mutex_);
    std::cout << "Record\n";
    op_idx_ = op_idx;
    op_name_ = op_name;
    op_idx_2_op_name_[op_idx_] = op_name_;
    is_pause_ = false;
    cv_.notify_one();
    std::cout << "Record end\n";
}

int main() {
 for(int i = 0; i < 2; i++) {
  TraceGPUMemoTool tool(1);
  for(int j = 0; j < 10; j++){
    tool.Record(j, "conv");
    std::cout << "run op j: " << j << std::endl;
    std::this_thread::sleep_for(std::chrono::microseconds(10000));
    tool.Pause();
  }
  tool.Stop();
 }
}