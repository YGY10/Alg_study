#include <iostream>
#include <memory>

struct MyStruct {
    int x;
    MyStruct(int val) : x(val){
        std::cout << "MyStruct created with value: " << x << std::endl;
    }
    ~MyStruct(){
        std::cout << "MyStruct value " << x << " deleate" << std::endl;
    }
};

int main () {
    // auto p = std::make_shared<MyStruct>(5);
    // std::cout << "After creation of the shared pointer p:" << std::endl;
    /* 下面这是输出
     MyStruct created with value: 5
     After creation of the shared pointer p:
     MyStruct value 5 deleate*/

     int * ptr1 = new int(5);
     int * ptr2 = nullptr;
     std::cout << "bool ptr1: " << (ptr1 != nullptr) << std::endl;
     std::cout << "bool ptr2: " << (ptr2 != nullptr) << std::endl;
}