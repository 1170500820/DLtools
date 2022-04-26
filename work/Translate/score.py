import os


def main():
    subfiles = os.listdir('.')
    available_files = []
    for elem in subfiles:
        if elem[-12:] == '-sample.json':
            available_files.append(elem)

if __name__ == '__main__':
    main()