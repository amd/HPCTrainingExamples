#!/usr/bin/env python3
"""Generate a list of UNIQUE email addresses, one per line on stdout.

Emails are unique per recipient (guarantee #2 for a perfect hash) but live in a
huge, sparse string space (they fail guarantee #1 -- a dense bounded range), so
they are the canonical input for a COMPACT hash, not a dense perfect-hash scatter.
Uniqueness is enforced here by appending a disambiguating counter when a
first.last@domain would repeat.

Usage:  python3 gen_emails.py [count] [seed]
        python3 gen_emails.py 200000 > emails.txt
"""
import sys, random

FIRST = ["james","mary","john","patricia","robert","jennifer","michael","linda",
         "william","elizabeth","david","barbara","richard","susan","joseph",
         "jessica","thomas","sarah","charles","karen","chris","nancy","daniel",
         "lisa","matthew","betty","anthony","sandra","mark","ashley","paul","kim"]
LAST  = ["smith","johnson","williams","brown","jones","garcia","miller","davis",
         "rodriguez","martinez","hernandez","lopez","gonzalez","wilson","anderson",
         "thomas","taylor","moore","jackson","martin","lee","perez","thompson",
         "white","harris","sanchez","clark","ramirez","lewis","robinson","walker"]
DOMAIN = ["example.com","mail.org","inbox.net","fastmail.io","company.co",
          "univ.edu","service.gov","shop.store","news.media","cloud.dev"]

def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    seed  = int(sys.argv[2]) if len(sys.argv) > 2 else 20260908
    random.seed(seed)
    seen = {}
    out = []
    while len(out) < count:
        f = random.choice(FIRST); l = random.choice(LAST); d = random.choice(DOMAIN)
        local = f + "." + l
        k = local + "@" + d
        c = seen.get(k, 0)
        seen[k] = c + 1
        if c:                       # collision -> disambiguate to keep uniqueness
            k = local + str(c) + "@" + d
        out.append(k)
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
