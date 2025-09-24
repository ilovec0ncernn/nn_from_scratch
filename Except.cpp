#include "Except.h"

#include <exception>
#include <iostream>

namespace except {

void react() {
    try {
        throw;
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    } catch (...) {
    }
}

}  // namespace except