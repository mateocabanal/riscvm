void print_str(char *msg, int len) {
  asm("addi a7, x0, 64\n"
      "add a0, zero, 1\n"
      "add a1, zero, %[ptr]\n"
      "add a2, x0, %1\n"
      "ecall" ::[ptr] "r"(msg),
      "r"(len));
}

int my_mul(int a, int b) {
  int result = 0;
  int sign = 1;

  // Handle the sign of the result
  if (a < 0) {
    a = -a;       // Make 'a' positive
    sign = -sign; // Flip the sign
  }
  if (b < 0) {
    b = -b;       // Make 'b' positive
    sign = -sign; // Flip the sign
  }

  // Optimize by making 'a' the smaller number to reduce loop iterations
  if (a > b) {
    int temp = a;
    a = b;
    b = temp;
  }

  // Add 'b' to result 'a' times
  for (int i = 0; i < a; i++) {
    result += b;
  }

  // Apply the sign to the result without using '*'
  if (sign < 0) {
    result = -result;
  }

  return result;
}

int my_div(int numerator, int denominator) {
  int quotient = 0;
  int sign = (numerator < 0) ^ (denominator < 0) ? -1 : 1;

  numerator = numerator < 0 ? -numerator : numerator;
  denominator = denominator < 0 ? -denominator : denominator;

  while (numerator >= denominator) {
    numerator -= denominator;
    quotient++;
  }

  return my_mul(sign, quotient);
}

// modulus function without using the standard library
int my_mod(int numerator, int denominator) {
  int sign = numerator < 0 ? -1 : 1;

  numerator = numerator < 0 ? -numerator : numerator;
  denominator = denominator < 0 ? -denominator : denominator;

  while (numerator >= denominator) {
    numerator -= denominator;
  }

  return my_mul(sign, numerator);
}

int reverse_int(int num) {
  int reversed = 0;

  while (num != 0) {
    int digit = my_mod(num, 10); // Extract the last digit
    reversed =
        my_mul(reversed, 10) + digit; // Append the digit to reversed number
    num = my_div(num, 10);            // Remove the last digit from num
  }

  return reversed;
}

void print_int(int num) {
  int reversed = reverse_int(num);

  while (reversed > 0) {
    char buf = (my_mod(reversed, 10)) + 48;
    print_str(&buf, 1);
    reversed = my_div(reversed, 10);
  }
}

void exit(int code) {
  asm("addi a7, x0, 93\n"
      "add a0, zero, %[code]\n"
      "ecall" ::[code] "r"(code));
  __builtin_unreachable();
}

unsigned int fib(int n) {
  int t1 = 0, t2 = 1, nextTerm = 1;
  for (int i = 3; i <= n; ++i) {
    print_int(nextTerm);

    if (i != n)
      print_str(", ", 2);

    t1 = t2;
    t2 = nextTerm;
    nextTerm = t1 + t2;
  }
  print_str("\n", 1);
}

void _start() {
  char *msg = "Hello, World\n";
  print_str(msg, 13);
  fib(35);
  exit(0);
}
