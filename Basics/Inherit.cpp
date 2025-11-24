#include <iostream>

class Base {
   public:
    virtual float Select(float time, float energy) = 0;
    void Talk() { std::cout << " Talk HHH " << std::endl; }
};

class Human : public Base {
   public:
    float Select(float time, float energy) override {
        std::cout << "Human Select" << std::endl;
        return 0.f;
    };
    int Select(int time, float energy) {
        std::cout << "Human Select Int" << std::endl;
        Talk();
        return 0;
    };
};

int main() {
    Human h1;
    h1.Select(1, 0.3f);
}