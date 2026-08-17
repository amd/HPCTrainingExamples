#!/usr/bin/env python3
# Copyright AMD 2026, MIT License
# Author: Bob Robey Bob.Robey@amd.com with AI tool help
"""Generate a sample last-name list whose first-letter distribution matches
approximate U.S. surname-initial frequencies. One surname per line on stdout.

Usage:  python3 gen_names.py [count] [seed]
        python3 gen_names.py 5000 > names.txt
"""
import sys, random

# Approximate share (%) of U.S. population by surname initial. Rounded; the
# generator normalizes these into sampling weights, so exact sums don't matter.
FREQ = {
    'A': 3.0, 'B': 9.5, 'C': 8.0, 'D': 5.5, 'E': 2.0, 'F': 3.5, 'G': 6.0,
    'H': 7.0, 'I': 0.5, 'J': 2.5, 'K': 4.0, 'L': 5.0, 'M': 9.5, 'N': 2.0,
    'O': 1.5, 'P': 5.0, 'Q': 0.2, 'R': 6.0, 'S': 10.0, 'T': 4.0, 'U': 0.2,
    'V': 1.5, 'W': 5.0, 'X': 0.1, 'Y': 0.5, 'Z': 0.7,
}

# Small pools of real, ASCII-only common surnames per initial. Duplicates in the
# output are realistic (many Smiths) and exercise the compact-hash probing.
POOL = {
    'A': ["Adams","Allen","Anderson","Alvarez","Aguilar","Austin"],
    'B': ["Brown","Baker","Brooks","Bennett","Bell","Bailey","Butler","Barnes"],
    'C': ["Clark","Campbell","Carter","Collins","Cooper","Cook","Cox","Chavez"],
    'D': ["Davis","Diaz","Dixon","Dunn","Duncan","Douglas"],
    'E': ["Evans","Edwards","Ellis","Estrada","Elliott"],
    'F': ["Foster","Fisher","Flores","Ford","Freeman","Fox"],
    'G': ["Garcia","Green","Gonzalez","Gray","Gomez","Griffin"],
    'H': ["Harris","Hall","Hill","Howard","Hughes","Hernandez","Henderson"],
    'I': ["Ingram","Irwin","Ibarra"],
    'J': ["Johnson","Jones","Jackson","Jenkins","James"],
    'K': ["King","Kelly","Kim","Knight","Kennedy"],
    'L': ["Lee","Lewis","Long","Lopez","Lynch"],
    'M': ["Miller","Martin","Moore","Martinez","Mitchell","Morgan","Murphy","Morris"],
    'N': ["Nelson","Nguyen","Newman","Nichols"],
    'O': ["Olson","Owens","Ortiz","Oliver"],
    'P': ["Parker","Perez","Peterson","Powell","Price","Patterson"],
    'Q': ["Quinn","Quintero"],
    'R': ["Robinson","Rodriguez","Roberts","Reed","Ramirez","Reyes","Russell"],
    'S': ["Smith","Scott","Stewart","Sanchez","Sanders","Simmons","Sullivan","Stevens"],
    'T': ["Taylor","Thomas","Thompson","Turner","Torres","Tucker"],
    'U': ["Underwood","Ubarri","Upton"],
    'V': ["Vargas","Vasquez","Vaughn","Vincent"],
    'W': ["Williams","Wilson","White","Wright","Wood","Ward","Watson","Walker"],
    'X': ["Xiong","Xu"],
    'Y': ["Young","Yang","Yates"],
    'Z': ["Zimmerman","Zuniga","Zhang"],
}

def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260908
    random.seed(seed)
    letters = list(FREQ.keys())
    weights = [FREQ[c] for c in letters]
    out = []
    for _ in range(count):
        c = random.choices(letters, weights=weights, k=1)[0]
        out.append(random.choice(POOL[c]))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
