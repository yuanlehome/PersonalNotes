#include "addressbook.pb.h"
#include <fstream>

int main () {

  // 构建一个 address_book
  tutorial::AddressBook my_address_book;

  // 往通信录中加入人员信息
  tutorial::Person* people = my_address_book.add_people();

  people->set_name("xiaoming");
  people->set_id(9827);
  people->set_email("123@qq.com");
  tutorial::Person_PhoneNumber* phone_number = people->add_phones();

  phone_number->set_number("1234567890");
  phone_number->set_type(tutorial::Person_PhoneType::Person_PhoneType_MOBILE);
  
  // 将构建好的通信录序列化保存在磁盘中
  std::ofstream outFile("ADDRESS_BOOK", std::ios::out | std::ios::binary);
  my_address_book.SerializeToOstream(&outFile);
  outFile.close();

  //将磁盘中的文件反序列化到 address_book 中
  tutorial::AddressBook address_book;
  std::ifstream inFile("ADDRESS_BOOK", std::ios::in | std::ios::binary);
  address_book.ParseFromIstream(&inFile);
  inFile.close();

  // 读取通信录中的人员信息
  auto &person = address_book.people(0);
  std::cout << person.name() << std::endl;
  std::cout << person.id() << std::endl;
  std::cout << person.email() << std::endl;
  
  auto &number = person.phones(0);
  std::cout << number.type() << std::endl;
  std::cout << number.number() << std::endl;
  return 0;
}