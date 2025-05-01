#!/bin/bash

if [ $# -ne 1 ]
then
	echo "Requires one argument with filename of markdown file"
	echo "Usage: ./markdown2pdf.sh test.md"
fi

if [ -z "$1" ]
then
	echo "Requires one argument with filename of markdown file"
	echo "Usage: ./markdown2pdf.sh test.md"
fi

if [ ! -f "$1" ]
then
	echo "file $1 does not exist"
	echo "Usage: ./markdown2pdf.sh test.md"
fi

filename="${1%.*}"
pandoc "$1" -V geometry:margin=1in --toc -s -o "$filename.pdf"
