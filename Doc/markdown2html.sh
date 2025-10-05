#!/bin/bash

if [ $# -ne 1 ]
then
	echo "Requires one argument with filename of markdown file"
	echo "Usage: ./markdown2html.sh event_name"
fi

if [ -z "$1" ]
then
	echo "Requires one argument with filename of markdown file"
	echo "Usage: ./markdown2html.sh event_name"
fi

if [ ! -f "$1.md" ]
then
	echo "file $1 does not exist"
	echo "Usage: ./markdown2html.sh event_name"
fi

filename="${1}"
EVENT_NAME=`echo ${1} | sed -e 's/_/ /g' `
pandoc "$filename.md" -V geometry:margin=1in --metadata title="$EVENT_NAME" --toc -s -o "$filename.html"
