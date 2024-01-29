def my_function():
    result = "Hello, World!"
    with open("log.txt", "a") as log_file:
        log_file.write(result + "\n")
    return result

my_function()