def greet():
    print("Hi There")
    print("Welcome aboard")

    greet()

def greet(first_name, last_name): # Input for parameter arguments are required
    print("Hi There")
    print("Welcome aboard")

greet("Mosh", "Hamedani") # Argument

def greet(first_name, last_name):
    print(f"Hi {first_name}{last_name}")
    print("Welcome aboard")

greet("Mosh", "Hamedani")
greet("John", "Smith")

def greet(name):
    print(f"Hi {name}")

def get_greeting(name):
    return f"Hi {name}"

message = get_greeting("Mosh")
print(message)
file = open("content.txt", "w")
file.write(message)

def greet(name):
    print(f"Hi {name}")

print(greet("Mosh"))


def increment(number, by):
    return number + by

result = increment(2, 1)
print(result)

def multiply(*numbers):
    total = 1
    for number in numbers:
        total = total * number
        total *= number
        print(number)
    return total

print(multiply(2, 3, 4, 5))

def save_user(**user):
    print(user)

save_user(id=1, name="John", age=22)

def greet(name):
    message = "a"

def send_email(name):
    global  message
    message = "b"

greet("Mosh")
print(message)

def fizz_buzz(input):

print(fizz_buzz(5))
