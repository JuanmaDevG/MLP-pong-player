// Esta clase controla el desarrollo de 1 partido

#ifndef __TPARTIDO_H_
#define __TPARTIDO_H_

#include "StdPijo.h"
#include "TCampo.h"
#include "TMarcador.h"
#include "TEtiqueta.h"
#include "TMensaje.h"
#include <time.h>
#include <stdlib.h>
#include <string>

// Maximo nivel implementado
#define _MAX_NIVEL_  5 

// Para establecer los controles
#define HUMANO       true
#define MAQUINA      false

class TPartido
{
public:
   // Constructor y destructor
   TPartido  ();
   ~TPartido ();
   
   // Funciones Set
   void PonControles (bool pctrlIzq, bool pctrlDcha);
   void PonCampo     (int NumCampo);
   void PonNivel     (int pnivel);
   void PonIA        (unsigned izq, unsigned der);
   void PonMaxTiempo (int pmaxtiempo);
   void PonMaxGoles  (int pmaxgoles);
   void PonFPS       (double pFPS);
   
   // Desarrollo
   int  Jugar        ();
   
   // Funciones Get
   bool HumanoIzq    () const;
   bool HumanoDcha   () const;
   int  Campo        () const;
   int  Nivel        () const;
   unsigned maxIA    () const;

   // Leer de fichero
   void LeerPesosIAs(std::string const& fichero);
   void IniciarInteligencias (std::string const& fichero);
private:
   TCampo    campo;           // Campo de juego
   TMarcador marcador;        // Marcador del partido
   bool   ctrlIzq, ctrlDcha;  // Control de los lados (true=human)
   int    numCampo;           // Numero del campo de juego actual
   int    nivel;              // Nivel de dificultad
   int    maxtiempo;          // Tiempo maximo de juego
   int    maxgoles;           // Maximo numero de goles
   double FPS;                // Fotogramas por segundo

   unsigned IA_Izq;           // Inteligencia artificial del lado izquierdo
   unsigned IA_Der;           // Inteligencia artificial del lado derecho
 
   void   EfectoGol(int lado);// Realiza el efecto de gol
   void   Saque (int Dir);    // Realiza el saque de centro
   double Inteligencia(int j);// Inteligencia artificial xD
};

#endif
