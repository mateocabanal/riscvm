#include <iostream>
#include <string>

class StringSet {
private:
  std::string *arr;
  int capacity;
  int current_size = 0;

public:
  StringSet();

  StringSet(const StringSet &call_object);

  ~StringSet();
  std::string tostring();

  StringSet &operator=(const StringSet &call_object);
  bool insert(const std::string &str_insert);
  void remove(const std::string &str_insert);
  int find(const std::string &str_insert) const;
  int size() const;

  StringSet &ssUnion(const StringSet &call_object);
  StringSet &intersection(const StringSet &call_object);
  StringSet &difference(const StringSet &call_object);
};

std::string StringSet::tostring() {
  std::string str = "";
  for (int i = 0; i < current_size; i++) {
    str += arr[i] + " ";
  }
  str += "\ncapacity: ";
  str += std::to_string(capacity);
  str += "\ncurrent_size: ";
  str += std::to_string(current_size);
  return str;
}
StringSet::StringSet() {
  capacity = 2;
  arr = new std::string[capacity];
}
StringSet::StringSet(const StringSet &call_object) {
  capacity = call_object.capacity;
  current_size = call_object.current_size;
  arr = new std::string[capacity];
  for (int i = 0; i < current_size; i++) {
    arr[i] = call_object.arr[i];
  }
}
StringSet::~StringSet() { delete[] arr; }
StringSet &StringSet::operator=(const StringSet &call_object) {
  if (this == &call_object) {
    return *this;
  }
  delete[] arr;
  arr = new std::string[call_object.capacity];
  capacity = call_object.capacity;
  current_size = call_object.current_size;
  for (int i = 0; i < current_size; i++) {
    arr[i] = call_object.arr[i];
  }
  return *this;
}
bool StringSet::insert(const std::string &str_insert) {
  for (int i = 0; i < current_size; i++) {
    if (arr[i] == str_insert) {
      return false;
    }
  }
  if (capacity == current_size) {
    int new_capacity = capacity * 2;
    std::string *new_arr = new std::string[new_capacity];
    for (int i = 0; i < current_size; i++) {
      new_arr[i] = arr[i];
    }
    delete[] arr;
    arr = new_arr;
    capacity = new_capacity;
  }

  arr[current_size++] = str_insert;
  return true;
}
void StringSet::remove(const std::string &str_insert) {
  for (int i = 0; i < current_size; i++) {
    if (arr[i] == str_insert) {
      std::string place = arr[i];
      arr[i] = arr[current_size - 1];
      arr[current_size - 1] = place;
      current_size--;
    }
  }
}
int StringSet::find(const std::string &str_insert) const {
  for (int i = 0; i < current_size; i++) {
    if (arr[i] == str_insert) {
      return i;
    }
  }
  return -1;
}
int StringSet::size() const { return current_size; }
StringSet &StringSet::ssUnion(const StringSet &call_object) {
  StringSet *new_set = new StringSet();
  for (int i = 0; i < current_size; i++) {
    new_set->insert(arr[i]);
  }
  for (int i = 0; i < call_object.current_size; i++) {
    new_set->insert(call_object.arr[i]);
  }
  return *new_set;
}
StringSet &StringSet::intersection(const StringSet &call_object) {
  StringSet *new_set = new StringSet();
  for (int i = 0; i < current_size; i++) {
    for (int j = 0; j < call_object.current_size; j++) {
      if (arr[i] == call_object.arr[j]) {
        new_set->insert(arr[i]);
      }
    }
  }
  return *new_set;
}
StringSet &StringSet::difference(const StringSet &call_object) {
  StringSet *new_set = new StringSet();
  for (int i = 0; i < current_size; i++) {
    bool found = false;
    for (int j = 0; j < call_object.current_size; j++) {
      if (arr[i] == call_object.arr[j]) {
        found = true;
        break;
      }
    }
    if (!found) {
      new_set->insert(arr[i]);
    }
  }
  return *new_set;
}

using std::cout, std::endl;
void basicTest() {
  StringSet sset1;
  sset1.insert("cat");
  sset1.insert("bat");
  sset1.insert("rat");
  bool insertTest = sset1.insert("badger");
  cout << sset1.tostring() << endl;
  cout << "insert success: " << insertTest << endl;

  StringSet sset2;
  sset2.insert("elephant");
  sset2.insert("bat");
  sset2.insert("hamster");
  sset2.insert("weasel");
  sset2.remove("weasel");
  cout << sset2.tostring() << endl;
  int findTest = sset2.find("weasel");
  cout << "index of weasel = " << findTest << endl;

  // Use the copy constructor to build a StringSet with sset1 U sset2
  StringSet sset3(sset1.ssUnion(sset2));
  cout << sset3.tostring() << endl;

  // Use overloaded assignment operator to make a StringSet with sset2 int sset1
  StringSet sset4;
  sset4 = sset2.intersection(sset1);
  cout << sset4.tostring() << endl;

  // Use overloaded assignment operator to make a StringSet with sset2 - sset1
  sset4 = sset2.difference(sset1);
  cout << sset4.tostring() << endl;
  cout << "end basic test" << endl;
}
int main() {
  basicTest();
  return 0;
}
