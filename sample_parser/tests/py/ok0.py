# Simple assignments
x = 10
y = 20

# Expression evaluation
z = x + y * 2

# If-elif-else with multi-line blocks and inline suites
if z > 50:
    result = "Large"
    size = "Big"
elif z > 30: result = "Medium"; size = "Average"
else:
    result = "Small"
    size = "Tiny"

# While loop with multi-line block and inline suite
count = 0
while count < 3:
    count = count + 1
    print("Looping:", count)
else: final_count = count; print("Loop done")

# Try-except-finally with inline suite
try:
    a = 1 / 0
except ZeroDivisionError: a = "Error"; print("Caught exception")
finally:
    message = "Handled"
    print("Cleanup done")

# Print results
print(result, size)
print(final_count, a, message)