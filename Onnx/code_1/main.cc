
// g++ main.cc addressbook.pb.cc -static -lprotobuf

#include "onnx.pb.h"
#include <fstream>
#include <iostream>

#define COUT(x) std::cout<< x << std::endl;

void display_graph(const onnx::GraphProto& graph) {
  // 打印 Graph
  std::cout << "\n***************** graph *****************" << std::endl;
  std::cout << "graph.name: " << graph.name() << std::endl;
  std::cout << "graph.doc_string: " << graph.doc_string() << std::endl;
  std::cout << "graph.input: ";
  for(int i = 0; i < graph.input_size(); i++) {
    std::cout << graph.input(i).name() << " ";
  }
  std::cout << std::endl;
  std::cout << "graph.output: ";
  for(int i = 0; i < graph.output_size(); i++) {
    std::cout << graph.output(i).name() << " ";
  }
  std::cout << std::endl;
  std::cout << "graph.value_info: ";
  for(int i = 0; i < graph.value_info_size(); i++) {
    std::cout << graph.value_info(i).name() << " ";
  }
  std::cout << std::endl;
  std::cout << "graph.initializer: ";
  for(int i = 0; i < graph.initializer_size(); i++) {
    std::cout << graph.initializer(i).name() << " ";
  }
  std::cout << std::endl;

  std::cout << "\n***************** nodes *****************" << std::endl;
  std::cout << "graph.node_size: " << graph.node_size() << std::endl;
  int node_size = graph.node_size();
  for(int i = 0; i < node_size; i++) {
    auto &node = graph.node(i);
    std::cout << "{" << std::endl;
    std::cout << "  op_type: " << node.op_type() << std::endl; 
    std::cout << "  input:   "; 
    for(int j = 0; j < node.input_size(); j++) {
      std::cout << node.input(j) << " ";
    }
    std::cout << std::endl;
    std::cout << "  output:  "; 
    for(int j = 0; j < node.output_size(); j++) {
      std::cout << node.output(j) << " ";
    }
    std::cout << std::endl;
    std::cout << "  attribute: {" << std::endl; 
    for(int j = 0; j < node.attribute_size(); j++) {
      std::cout << "\t       " << node.attribute(j).name() << " :";
      if(node.attribute(j).type() == onnx::AttributeProto::GRAPH) {
        auto& sub_graph = node.attribute(j).g();
        display_graph(sub_graph);
      }
      std::cout << std::endl;
    }
    std::cout << "\t     }" << std::endl;
    std::cout << "}" << std::endl;
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "Usage:  <onnx_model.onnx>" << std::endl;
    return -1;
  }
  std::string model_file = argv[1];

  // 加载模型
  onnx::ModelProto onnx_model;
  std::ifstream inFile(model_file, std::ios::in | std::ios::binary);
  onnx_model.ParseFromIstream(&inFile);
  inFile.close();

  // 打印 ModelProto 中的一些信息
  std::cout << "ir_version: " << onnx_model.ir_version() << std::endl;
  std::cout << "opset_import_size: " << onnx_model.opset_import_size() << std::endl;
  std::cout << "OperatorSetIdProto domain: " << onnx_model.opset_import(0).domain() << std::endl;
  std::cout << "OperatorSetIdProto version: " << onnx_model.opset_import(0).version() << std::endl;
  std::cout << "producer_name: " << onnx_model.producer_name() << std::endl;
  std::cout << "producer_version: " << onnx_model.producer_version() << std::endl;
  std::cout << "domain: " << onnx_model.domain() << std::endl;
  std::cout << "model_version: " << onnx_model.model_version() << std::endl;
  std::cout << "doc_string: " << onnx_model.doc_string() << std::endl;
  std::cout << "metadata_props_size: " << onnx_model.metadata_props_size() << std::endl;
  if(onnx_model.metadata_props_size()) {
    std::cout << "StringStringEntryProto key: " << onnx_model.metadata_props(0).key() << std::endl;
    std::cout << "StringStringEntryProto value: " << onnx_model.metadata_props(0).value() << std::endl;
  }
  std::cout << "functions_size: " << onnx_model.functions_size() << std::endl;

  auto &graph = onnx_model.graph();
  display_graph(graph);
  return 0;
}