#!/bin/bash

for filename in *.docx; do
    filename=$(printf %q "${filename%%.*}")
    echo $filename
    pandoc -f docx -t markdown "$filename".docx -o "$filename".md --extract-media media/$filename/
    mv media/$filename/media/* media/$filename
    rmdir media/$filename/media 
done
