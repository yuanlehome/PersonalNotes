import addressbook_pb2

# 构建一个 address_book
my_address_book = addressbook_pb2.AddressBook()

# 往通信录中加入人员信息
people = my_address_book.people.add()
people.name = "xiaoming"
people.id = 9827
people.email = "123@qq.com"
  
phone = people.phones.add()
phone.number = "1234567890"
phone.type = addressbook_pb2.Person.HOME

# 将构建好的通信录序列化保存在磁盘中
f = open("ADDRESS_BOOK", "wb")
f.write(my_address_book.SerializeToString())
f.close()

# 将磁盘中的文件反序列化到 address_book 中
f = open("ADDRESS_BOOK", "rb")
address_book = addressbook_pb2.AddressBook()
address_book.ParseFromString(f.read())
f.close()

# 读取通信录中的人员信息
person = address_book.people[0]
print(person.name)
print(person.id)
print(person.email)
  
number = person.phones[0]
print(number.type)
print(number.number)

