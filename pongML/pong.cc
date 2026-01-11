#include "StdPijo.h"
#include "TPartido.h"
#include "TMenu.h"
#include <string>
#include <vector>
#include <sstream>
#include <tuple>
#include <signal.h>

// Fotogramas por segundo
#define FPS  35

// Inteligencias de Machine Learning como parámetro
unsigned IA_Izq = 0;
unsigned IA_Der = 0;

// Funcion para terminar la ejecucion y 
// dejar el terminal en condiciones
static void
finish (int sig)
{
   STDP::BorraPantalla();
   STDP::Refrescar();
   STDP::Terminar();
   exit (0);
}

// Presentacion
static void
Presentacion (double fps)
{
   #define _      0
   #define TPU    0.05

   double t;
   const int Ancho        = 32;
   const int Alto         = 11;
   const int Caracter[]   = { ' ', ' ', '*', ' ' };
   const int Atributo[]   = { STDP_A_NORMAL, STDP_A_NEGRITA, 
                              STDP_A_NEGRITA, STDP_A_NORMAL };
   const int ColorTexto[] = { STDP_C_NEGRO,  STDP_C_NEGRO, 
                              STDP_C_AZUL, STDP_C_NEGRO };
   const int ColorFondo[] = { STDP_C_VERDE,  STDP_C_AMARILLO, 
                              STDP_C_ROJO, STDP_C_CIAN };
   const char Dibujo[Alto][Ancho]  =
      {  { 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2 },
         { 2,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,2 },
         { 2,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,2 },
         { 2,_,_,1,1,1,1,1,_,_,1,1,1,1,1,_,_,1,_,_,_,1,_,_,1,1,1,1,1,_,_,2 }, 
         { 2,_,_,1,3,3,3,1,3,_,1,3,3,3,1,3,_,1,1,_,_,1,3,_,1,3,3,3,3,3,_,2 },
         { 2,_,_,1,1,1,1,1,3,_,1,3,_,_,1,3,_,1,3,1,_,1,3,_,1,3,1,1,1,_,_,2 },
         { 2,_,_,1,3,3,3,3,3,_,1,3,_,_,1,3,_,1,3,_,1,1,3,_,1,3,_,3,1,3,_,2 },
         { 2,_,_,1,3,_,_,_,_,_,1,1,1,1,1,3,_,1,3,_,_,1,3,_,1,1,1,1,1,3,_,2 },
         { 2,_,_,_,3,_,_,_,_,_,_,3,3,3,3,3,_,_,3,_,_,_,3,_,_,3,3,3,3,3,_,2 },
         { 2,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,2 },
         { 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2 }  };

   const int  posX = (STDP::TamX() - Ancho) / 2;
   const int  posY = 1;

   // Borramos la pantalla
   STDP::BorraPantalla ();

   // Formacion del dibujo
   for (int finX = posX; finX < posX + Ancho; finX++)
   {
      for (int finY = posY; finY < posY + Alto; finY++)
      {
         int lado = rand() % 4;
         int car  = Dibujo[finY-posY][finX-posX];
         int ch   = Caracter   [car];
         int Atr  = Atributo   [car];
         int CT   = ColorTexto [car];
         int CF   = ColorFondo [car];
         float x  = (float)(rand() % STDP::TamX()), vx = 0.0;
         float y  = (float)(rand() % STDP::TamY()), vy = 0.0;
         int ix, iy;
         bool Fin;

         // tomamos el tiempo inicial
         t = clock () + (1.0/fps) * CLOCKS_PER_SEC;

         // El trozo de dibujo sale de uno de los 4 lados de la pantalla
         switch(lado)
         {
            case 0:  { x = 0.0; break; }
            case 1:  { y = 0.0; break; }
            case 2:  { x = (float)(STDP::TamX() - 1); break; }
            default: { y = (float)(STDP::TamY() - 1); break; }
         }
         
         // Ahora calculamos el vector velocidad y la posicion entera
         vx = (((float)finX) - x) / ((int)(fps * TPU));
         vy = (((float)finY) - y) / ((int)(fps * TPU));
         ix  = (int)(x + 0.5);
         iy  = (int)(y + 0.5);

         // Y ahora desplazamos el trozo
         Fin = false;
         do
         {
            // Dibujamos
            STDP::PonCursor   (ix, iy);
            STDP::CambiaColor (Atr, CT, CF);
            STDP::EscribeCh   (ch);
            STDP::Refrescar ();

            // Esperamos al clock y tomamos el tiempo
            while (t > clock());
            t = clock() + (1.0/fps) * CLOCKS_PER_SEC;

            // Miramos a ver si hemos llegado al final
            if (ix != finX || iy != finY)
            {   
               // Borramos
               STDP::PonCursor(ix,iy);
               if ((ix>=posX && ix<finX && iy>=posY && iy<posY+Alto) ||
                   (ix==finX && iy>=posY && iy<finY))
               {
                  unsigned car1 = static_cast<unsigned>(Dibujo[iy-posY][ix-posX]);
                  const int  ch1  = Caracter[car1];
                  const int  Atr1 = Atributo[car1];
                  const int  CT1  = ColorTexto[car1];
                  const int  CF1  = ColorFondo[car1];
                  STDP::CambiaColor (Atr1,CT1,CF1);
                  STDP::EscribeCh   (ch1); 
               }
               else
               {
                  STDP::CambiaColor (STDP_A_NORMAL,STDP_C_NEGRO,STDP_C_NEGRO);
                  STDP::EscribeCh   (' ');
               }

               // Calculamos nueva posicion
               x += vx; y += vy;
               ix = (int)(x + 0.5);
               iy = (int)(y + 0.5);
            }
            else 
               Fin = true;
            
            // Comprobacion de tecla, por si se desea finalizar
            if (STDP::ObtenTecla() == STDP_T_ENTER) return;
         } while (!Fin);
      }
   }

   // Ahora ponemos los nombres de los autores
   STDP::PonCursor   ((STDP::TamX()-15)/2, 13);
   STDP::CambiaColor (STDP_A_NEGRITA, STDP_C_BLANCO, STDP_C_NEGRO);
   STDP::EscribeStr  ("Realizado por :");

   STDP::PonCursor   ((STDP::TamX()-23)/2, 15);
   STDP::CambiaColor (STDP_A_NEGRITA, STDP_C_AZUL, STDP_C_NEGRO);
   STDP::EscribeStr  ("Eduardo Barredo Sanchez");

   STDP::PonCursor   ((STDP::TamX()-1)/2, 17);
   STDP::CambiaColor (STDP_A_NEGRITA, STDP_C_CIAN, STDP_C_NEGRO);
   STDP::EscribeStr  ("y");

   STDP::PonCursor   ((STDP::TamX()-27)/2, 19);
   STDP::CambiaColor (STDP_A_NEGRITA, STDP_C_AZUL, STDP_C_NEGRO);
   STDP::EscribeStr  ("Francisco Jose Gallego Duran");

   STDP::PonCursor   ((STDP::TamX()-27)/2, 23);
   STDP::CambiaColor (STDP_A_NORMAL, STDP_C_BLANCO, STDP_C_NEGRO);
   STDP::EscribeStr  ("Pulsa [ENTER] para continuar");

   while (STDP::ObtenTecla() != STDP_T_ENTER);
}

void
uso() {
   std::cerr 
   <<   "./pong <IA_Izq> <IA_Der> <segundos> <goles>"
   << "\n"
   << "\n   Ejecuta el juego seleccionando las IAs de los jugadores izquierda y derecha."
   << "\n   Ambas elecciones deben ser números enteros de 0 al máximo de IAs disponibles."
   << "\n";

   exit(-1);
}

std::string concatStrings(const std::vector<std::string>& err) {
   std::stringstream s;
   for (auto& e : err) s << e;
   return s.str();
}

void error(const std::vector<std::string>& err) {
   std::cerr << "ERROR: " << concatStrings(err) << "\n";
   finish (0);
   exit(-1);
}

void error(auto const&... val) {
   (std::cerr << ... << val); 
   std::cerr << '\n';
   finish (0);
   exit(-1);
}

bool isValidDec(const std::string& str) {
   for (auto&& c : str) {
      if ( !std::isdigit(c) )
            return false;
   }
   return true;
}

auto
parametrosLineaDeComandos(int argc, char **argv) {
   if (argc != 5) uso();
   TPartido Match;
   std::string str_ia_iz(argv[1]);
   std::string str_ia_de(argv[2]);
   std::string str_seconds(argv[3]);
   std::string str_goals(argv[4]);
   if (!isValidDec(str_ia_iz)) error({"'", str_ia_iz, "' no es un número decimal válido." });
   if (!isValidDec(str_ia_de)) error({"'", str_ia_de, "' no es un número decimal válido." });
   if (!isValidDec(str_seconds)) error({"'", str_seconds, "' no es una duración en segundos válida." });
   if (!isValidDec(str_goals)) error({"'", str_goals, "' no es un número decimal de goles válido." });
   IA_Izq = std::stoi(str_ia_iz);
   IA_Der = std::stoi(str_ia_de);
   int duracion = std::stoi(str_seconds);
   int maxgoles = std::stoi(str_goals);
   if ( duracion < 10 ) error ({"'", str_seconds, "' debe ser superior a 10 segundos." });
   if ( maxgoles < 1 ) error ({"'", str_goals, "' debe ser superior a 1 gol." });

   return std::tuple<int,int>{duracion, maxgoles};
}

// Programa principal
int 
main (int argc, char **argv)
{
   int      campo = 0, maxGoles = 3, level = 0,
            maxTiempo = 90, opcion;
   bool     ctrlA = MAQUINA, ctrlB = MAQUINA;
   int      Ancho = STDP::TamX();
   
   auto [mt, mg] = parametrosLineaDeComandos(argc, argv);
   maxTiempo = mt; maxGoles = mg;

   STDP::Inicializar();
   TPartido Match;
   TMenu    MenuPr(7), MenuTmp(5), MenuGls(5), 
            MenuCtr(4), MenuCmp(4), MenuNiv(5);

   Match.IniciarInteligencias("input.data");
   if ( IA_Izq > Match.maxIA() ) error ("'", IA_Izq, "' es mayor que la máxima IA disponible.");
   if ( IA_Der > Match.maxIA() ) error ("'", IA_Der, "' es mayor que la máxima IA disponible.");


   signal (SIGINT, finish);
   Presentacion(5 * FPS);

   // Configuramos los menus
   MenuPr.PonCadenaTitulo ("*-* MeNu PrInCiPaL *-*");
   MenuPr.PonCadenaOpcion (0, "Jugar");
   MenuPr.PonCadenaOpcion (1, "Controles");
   MenuPr.PonCadenaOpcion (2, "Dificultad");
   MenuPr.PonCadenaOpcion (3, "Tiempo Maximo");
   MenuPr.PonCadenaOpcion (4, "Goles Limite");
   MenuPr.PonCadenaOpcion (5, "Elegir Campo");
   MenuPr.PonCadenaOpcion (6, "Salir");
   MenuPr.PonFondo        (TMENU_MODO_PLANO, STDP_C_VERDE);
   MenuPr.PonAtrTitulo    (STDP_A_NORMAL, STDP_C_NEGRO, STDP_C_BLANCO); 
   MenuPr.PonMarco        (" Menu Principal ", 5, STDP_A_NEGRITA, STDP_C_VERDE, STDP_C_AMARILLO);
   MenuPr.PonPosTitulo    ((Ancho - 22)/2, 4);
   MenuPr.PonPosOpcion    (0, (Ancho - 5)/2, 7);
   MenuPr.PonPosOpcion    (1, (Ancho - 9)/2, 9);
   MenuPr.PonPosOpcion    (2, (Ancho - 10)/2, 11);
   MenuPr.PonPosOpcion    (3, (Ancho - 13)/2, 13);
   MenuPr.PonPosOpcion    (4, (Ancho - 12)/2, 15);
   MenuPr.PonPosOpcion    (5, (Ancho - 12)/2, 17);
   MenuPr.PonPosOpcion    (6, (Ancho - 5)/2, 19);

   MenuCtr.PonCadenaTitulo ("<-> CaMbIaR CoNtRoLeS <->");
   MenuCtr.PonCadenaOpcion (0, "HUMANO, HUMANO");
   MenuCtr.PonCadenaOpcion (1, "HUMANO, MAQUINA");
   MenuCtr.PonCadenaOpcion (2, "MAQUINA, HUMANO");
   MenuCtr.PonCadenaOpcion (3, "MAQUINA, MAQUINA");
   MenuCtr.PonFondo        (TMENU_MODO_PLANO, STDP_C_VERDE);
   MenuCtr.PonAtrTitulo    (STDP_A_NORMAL, STDP_C_MAGENTA, STDP_C_BLANCO); 
   MenuCtr.PonMarco        (" Cambiar Controles ", 5, STDP_A_NEGRITA, 
                            STDP_C_AMARILLO, STDP_C_AMARILLO);
   MenuCtr.PonPosTitulo    ((Ancho - 25)/2, 6);
   MenuCtr.PonPosOpcion    (0, (Ancho - 14)/2, 9);
   MenuCtr.PonPosOpcion    (1, (Ancho - 15)/2, 11);
   MenuCtr.PonPosOpcion    (2, (Ancho - 15)/2, 13);
   MenuCtr.PonPosOpcion    (3, (Ancho - 16)/2, 15);

   MenuTmp.PonCadenaTitulo ("<-> ElEgIr TiEmPo MaXiMo <->");
   MenuTmp.PonCadenaOpcion (0, "1 Minuto");
   MenuTmp.PonCadenaOpcion (1, " 2 Minutos");
   MenuTmp.PonCadenaOpcion (2, " 3 Minutos");
   MenuTmp.PonCadenaOpcion (3, " 5 Minutos");
   MenuTmp.PonCadenaOpcion (4, "Sin Limite de Tiempo");
   MenuTmp.PonFondo        (TMENU_MODO_PLANO, STDP_C_VERDE);
   MenuTmp.PonAtrTitulo    (STDP_A_NORMAL, STDP_C_MAGENTA, STDP_C_BLANCO); 
   MenuTmp.PonMarco        (" Elegir Tiempo Maximo ", 5, STDP_A_NEGRITA, 
                            STDP_C_AMARILLO, STDP_C_AMARILLO);
   MenuTmp.PonPosTitulo    ((Ancho - 28)/2, 4);
   MenuTmp.PonPosOpcion    (0, (Ancho - 8)/2, 7);
   MenuTmp.PonPosOpcion    (1, (Ancho - 10)/2, 9);
   MenuTmp.PonPosOpcion    (2, (Ancho - 10)/2, 11);
   MenuTmp.PonPosOpcion    (3, (Ancho - 10)/2, 13);
   MenuTmp.PonPosOpcion    (4, (Ancho - 20)/2, 15);

   MenuGls.PonCadenaTitulo ("<-> ElEgIr LiMiTe de GoLeS <->");
   MenuGls.PonCadenaOpcion (0, " 5 Goles");
   MenuGls.PonCadenaOpcion (1, "10 Goles");
   MenuGls.PonCadenaOpcion (2, "15 Goles");
   MenuGls.PonCadenaOpcion (3, "25 Goles");
   MenuGls.PonCadenaOpcion (4, "Sin Limite de Goles");
   MenuGls.PonFondo        (TMENU_MODO_PLANO, STDP_C_VERDE);
   MenuGls.PonAtrTitulo    (STDP_A_NORMAL, STDP_C_MAGENTA, STDP_C_BLANCO); 
   MenuGls.PonMarco        (" Elegir Limite de Goles ", 5, STDP_A_NEGRITA, 
                            STDP_C_AMARILLO, STDP_C_AMARILLO);
   MenuGls.PonPosTitulo    ((Ancho - 29)/2, 4);
   MenuGls.PonPosOpcion    (0, (Ancho - 8)/2, 7);
   MenuGls.PonPosOpcion    (1, (Ancho - 8)/2, 9);
   MenuGls.PonPosOpcion    (2, (Ancho - 8)/2, 11);
   MenuGls.PonPosOpcion    (3, (Ancho - 8)/2, 13);
   MenuGls.PonPosOpcion    (4, (Ancho - 19)/2, 15);

   MenuCmp.PonCadenaTitulo ("<-> ElEgIr CaMpO <->");
   MenuCmp.PonCadenaOpcion (0, "Campo Normal");
   MenuCmp.PonCadenaOpcion (1, "Campo XP");
   MenuCmp.PonCadenaOpcion (2, "Campo Reducido");
   MenuCmp.PonCadenaOpcion (3, "Campo Estrecho");
   MenuCmp.PonFondo        (TMENU_MODO_PLANO, STDP_C_VERDE);
   MenuCmp.PonAtrTitulo    (STDP_A_NORMAL, STDP_C_MAGENTA, STDP_C_BLANCO); 
   MenuCmp.PonMarco        (" Elegir Campo ", 5, STDP_A_NEGRITA, 
                            STDP_C_AMARILLO, STDP_C_AMARILLO);
   MenuCmp.PonPosTitulo    ((Ancho - 20)/2, 6);
   MenuCmp.PonPosOpcion    (0, (Ancho - 12)/2, 10);
   MenuCmp.PonPosOpcion    (1, (Ancho - 8)/2, 12);
   MenuCmp.PonPosOpcion    (2, (Ancho - 14)/2, 14);
   MenuCmp.PonPosOpcion    (3, (Ancho - 14)/2, 16);

   MenuNiv.PonCadenaTitulo ("<-> ElEgIr NiVeL de DiFiCuLtAd <->");
   MenuNiv.PonCadenaOpcion (0, "Soy un colador");
   MenuNiv.PonCadenaOpcion (1, "Juego como mi abuela");
   MenuNiv.PonCadenaOpcion (2, "Me he entrenado en el pasillo");
   MenuNiv.PonCadenaOpcion (3, "Soy un hacha");
   MenuNiv.PonCadenaOpcion (4, "Una pared no lo hace mejor");
   MenuNiv.PonFondo        (TMENU_MODO_PLANO, STDP_C_VERDE);
   MenuNiv.PonAtrTitulo    (STDP_A_NORMAL, STDP_C_MAGENTA, STDP_C_BLANCO); 
   MenuNiv.PonMarco        (" Elegir Nivel de Dificultad ", 5, STDP_A_NEGRITA, 
                            STDP_C_AMARILLO, STDP_C_AMARILLO);
   MenuNiv.PonPosTitulo    ((Ancho - 34)/2, 4);
   MenuNiv.PonPosOpcion    (0, (Ancho - 14)/2, 7);
   MenuNiv.PonPosOpcion    (1, (Ancho - 20)/2, 9);
   MenuNiv.PonPosOpcion    (2, (Ancho - 29)/2, 11);
   MenuNiv.PonPosOpcion    (3, (Ancho - 12)/2, 13);
   MenuNiv.PonPosOpcion    (4, (Ancho - 26)/2, 15);


   // Bucle principal
   do
   {
      opcion = MenuPr.Ejecutar(FPS);
      switch(opcion)
      {
         // Jugar
         case 0:
         {
            Match.PonMaxGoles  (maxGoles);  
            Match.PonMaxTiempo (maxTiempo);
            Match.PonCampo     (campo);
            Match.PonControles (ctrlA, ctrlB);
            Match.PonFPS       (FPS);
            Match.PonNivel     (level);
            Match.PonIA        (IA_Izq, IA_Der);
            Match.Jugar ();
            break;
         }
         // Cambiar Controles
         case 1:
         {
            switch (MenuCtr.Ejecutar(FPS))
            {
               case 0: { ctrlA =  HUMANO; ctrlB =  HUMANO; break; }
               case 1: { ctrlA =  HUMANO; ctrlB = MAQUINA; break; }
               case 2: { ctrlA = MAQUINA; ctrlB =  HUMANO; break; }
               case 3: { ctrlA = MAQUINA; ctrlB = MAQUINA; break; }
            }
            break;
         }
         // Nivel de dificultad
         case 2:
         {
            level = MenuNiv.Ejecutar(FPS);
            break;
         }
         // Elegir tiempo maximo
         case 3:
         {
            switch (MenuTmp.Ejecutar(FPS))
            {
               case 0: { maxTiempo =   60; break; }
               case 1: { maxTiempo =  120; break; }
               case 2: { maxTiempo =  180; break; }
               case 3: { maxTiempo =  300; break; }
               case 4: { maxTiempo =   -1; break; }
            }
            break;
         }
         // Elegir Limite de Goles
         case 4:
         {
            switch (MenuGls.Ejecutar(FPS))
            {
               case 0: { maxGoles =  5; break; }
               case 1: { maxGoles = 10; break; }
               case 2: { maxGoles = 15; break; }
               case 3: { maxGoles = 25; break; }
               case 4: { maxGoles = -1; break; }
            }
            break;
         }
         // Elegir Campo
         case 5:
         {
            campo = MenuCmp.Ejecutar(FPS);
            break;
         }
         default:
            break;
      }
   } while (opcion <= 5);

   finish (0);
   return 0;
}
