with open("dictionary.txt", "r") as file:
	words = file.readlines()
	words = [word[:-1].upper() for word in words]
	for word in words:
		for letter in word:
			if letter not in ["A","B","C","D","E","F"]:
				words.remove(word)
				break

with open("newdict.txt", "w") as nf:
	nf.writelines(words)


