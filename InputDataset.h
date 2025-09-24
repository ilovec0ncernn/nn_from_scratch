#include "Alias.h"

namespace nn {

struct Split {
    Matrix X_train, y_train;
    Matrix X_test, y_test;
};

class InputDataset {
   public:
    static Split load_mnist();
};

}  // namespace nn