COMP=g++
#CFLAGS=-g
CFLAGS=-O2 -std=c++23 -Wall 
OPTS=-lm -lncurses
OBJECTS=IA_ML.o TPaleta.o TPelota.o StdPijo.o TCampo.o TMarcador.o TEtiqueta.o TPartido.o TMenu.o TMensaje.o
EXE=pong

pong: $(OBJECTS) $(EXE).cc
	$(COMP) $(OBJECTS) $(OPTS) $(EXE).cc -o $(EXE) $(CFLAGS)

%.o: %.cc
	$(COMP) $^ -c -o $@ $(CFLAGS)

clean:
	rm -rf *.o *~

tar:
	tar cvzf ppoo.tgz *.cc *.h makefile readme.txt
