// 多态学习
// 意为相同的接口，不同的行为
// 用父类的指针/引用调用同一个函数，但最终行为由子类决定
// 小口诀：父类虚函数 子类重写 父类指针来调用
#include <iostream>

class Animal {
   public:
    // 虚函数的本质，是C++实现运行时多态的一种机制。它的核心原理原理是通过虚函数表和虚函数指针来动态决定调用哪个函数版本
    // vtable虚函数表：类级别的表，储存该类的虚函数地址列表
    // vptr虚函数指针: 每个含虚函数的对象里面都有一个指针，指向该类的vtable
    virtual void speak() { std::cout << "Animal speaks\n"; }
};

class Dog : public Animal {
   public:
    void speak() override { std::cout << "Dog barks\n"; }
};

// 类定义阶段，编译器为每个类生成一张vatable
// vtable_Animal: [0] -> Animal::speak
// vtable_Dog: [0] -> Dog::speak

int main() {
    Animal *a = new Dog();
    // new Dog()会分配一个Dog对象，其中包含一个隐藏的vptr指针：
    // vptr -> 指向 vtable_Dog
    a->speak();
}