SET /a r=15

for /l %%i in (1, 1, 9) do (java -jar negative-selection\negsel2.jar -self negative-selection\syscalls\snd-cert\N%r%\snd-cert.train -n %R% -r %%i -c -l <negative-selection\syscalls\snd-cert\N%r%\snd-cert.1.test > negative-selection\syscalls\snd-cert\N%r%\results\snd-cert.1.%%i.txt)
for /l %%i in (1, 1, 9) do (java -jar negative-selection\negsel2.jar -self negative-selection\syscalls\snd-cert\N%r%\snd-cert.train -n %R% -r %%i -c -l <negative-selection\syscalls\snd-cert\N%r%\snd-cert.2.test > negative-selection\syscalls\snd-cert\N%r%\results\snd-cert.2.%%i.txt)
for /l %%i in (1, 1, 9) do (java -jar negative-selection\negsel2.jar -self negative-selection\syscalls\snd-cert\N%r%\snd-cert.train -n %R% -r %%i -c -l <negative-selection\syscalls\snd-cert\N%r%\snd-cert.3.test > negative-selection\syscalls\snd-cert\N%r%\results\snd-cert.3.%%i.txt)

exit
REM start java -jar negative-selection\negsel2.jar -self negative-selection\syscalls\snd-cert\N%r%\snd-cert.train -n 15 -r 3 -c -l -g <negative-selection\syscalls\snd-cert\N15\snd-cert.1.test >negative-selection\syscalls\snd-cert\N15\results\snd-cert.1.3.txt >negative-selection\syscalls\snd-cert\N%r%\results\output.err
