#!/bin/bash
grep Z book-noms/*.txt >test.txt | sed -i 's\:* \\' test.txt
