SET /a N=15

for /l %%i in (2, 3, 11) do (java -jar negative-selection\negsel2.jar -self negative-selection\syscalls\snd-unm\N%N%\snd-unm.train -n %N% -r %%i -c -l <negative-selection\syscalls\snd-unm\N%N%\snd-unm.1.test > negative-selection\syscalls\snd-unm\N%N%\results\snd-unm.1.%%i.txt)
for /l %%i in (2, 3, 11) do (java -jar negative-selection\negsel2.jar -self negative-selection\syscalls\snd-unm\N%N%\snd-unm.train -n %N% -r %%i -c -l <negative-selection\syscalls\snd-unm\N%N%\snd-unm.2.test > negative-selection\syscalls\snd-unm\N%N%\results\snd-unm.2.%%i.txt)
for /l %%i in (2, 3, 11) do (java -jar negative-selection\negsel2.jar -self negative-selection\syscalls\snd-unm\N%N%\snd-unm.train -n %N% -r %%i -c -l <negative-selection\syscalls\snd-unm\N%N%\snd-unm.3.test > negative-selection\syscalls\snd-unm\N%N%\results\snd-unm.3.%%i.txt)

for /l %%i in (2, 3, 11) do (java -jar negative-selection\negsel2.jar -self negative-selection\syscalls\snd-cert\N%N%\snd-cert.train -n %N% -r %%i -c -l <negative-selection\syscalls\snd-cert\N%N%\snd-cert.1.test > negative-selection\syscalls\snd-cert\N%N%\results\snd-cert.1.%%i.txt)
for /l %%i in (2, 3, 11) do (java -jar negative-selection\negsel2.jar -self negative-selection\syscalls\snd-cert\N%N%\snd-cert.train -n %N% -r %%i -c -l <negative-selection\syscalls\snd-cert\N%N%\snd-cert.2.test > negative-selection\syscalls\snd-cert\N%N%\results\snd-cert.2.%%i.txt)
for /l %%i in (2, 3, 11) do (java -jar negative-selection\negsel2.jar -self negative-selection\syscalls\snd-cert\N%N%\snd-cert.train -n %N% -r %%i -c -l <negative-selection\syscalls\snd-cert\N%N%\snd-cert.3.test > negative-selection\syscalls\snd-cert\N%N%\results\snd-cert.3.%%i.txt)

exit
REM start java -jar negative-selection\negsel2.jar -self negative-selection\syscalls\snd-cert\N%r%\snd-cert.train -n 15 -r 3 -c -l -g <negative-selection\syscalls\snd-cert\N15\snd-cert.1.test >negative-selection\syscalls\snd-cert\N15\results\snd-cert.1.3.txt >negative-selection\syscalls\snd-cert\N%r%\results\output.err
