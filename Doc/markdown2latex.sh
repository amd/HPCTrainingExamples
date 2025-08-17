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
sed -i 's/╒/-/g' "$filename.md"
sed -i 's/╕/-/g' "$filename.md"
sed -i 's/═/-/g' "$filename.md"
sed -i 's/╤/-/g' "$filename.md"
sed -i 's/│/|/g' "$filename.md"
sed -i 's/╞/-/g' "$filename.md"
sed -i 's/╪/|/g' "$filename.md"
sed -i 's/╡/-/g' "$filename.md"
sed -i 's/╛/-/g' "$filename.md"
sed -i 's/╧/-/g' "$filename.md"
sed -i 's/╘/-/g' "$filename.md"
sed -i 's/┼/|/g' "$filename.md"
sed -i 's/─/-/g' "$filename.md"
sed -i 's/├/-/g' "$filename.md"
sed -i 's/┤/-/g' "$filename.md"
sed -i 's/μ/micro/g' "$filename.md"
sed -i 's/<img src="/![image](/g' "$filename.md"
sed -i 's|.png"/>|.png)|g' "$filename.md"
pandoc "$1" -V geometry:margin=1in --extract-media=. --toc -s -o "$filename.tex"


sed -i -e '/\\author{}/i\
\\usepackage{fancyvrb,newverbs,xcolor} \
\
\\definecolor{Light}{HTML}{F4F4F4} \
\
\\let\\oldtexttt\\texttt \
\\renewcommand{\\texttt}[1]{ \
  \\colorbox{Light}{\\oldtexttt{#1}} \
} \
\
\\let\\oldv\\verbatim \
\\let\\oldendv\\endverbatim \
\
\\def\\verbatim{\\par\\setbox0\\vbox\\bgroup\\oldv} \
\\def\\endverbatim{\\oldendv\\egroup\\fboxsep0pt \
\\noindent\\colorbox[gray]{0.8}{\\usebox0}\\par} \
\
\\usepackage{fancyvrb} \
' "$filename.tex"

sed -i 's/\\begin{verbatim}/\\begin{Verbatim}[fontsize=\\small]/' "$filename.tex"

sed -i 's/\\end{verbatim}/\\end{Verbatim}/' "$filename.tex"

