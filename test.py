class Person:
    count = 0

    def __init__(self, name):
        self.name = name
        self.count += 1

    @classmethod
    def get_instance_count(cls):
        return cls.count

    def get_name(self):
        return self.name

# Creating an instance
person1 = Person("Alice")

# Using instance method to access instance-specific attribute
print(person1.get_name())  # Output: Alice

# Using classmethod to access class-level variable
print(Person.get_instance_count())  # Output: 1
