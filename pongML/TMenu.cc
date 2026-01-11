#include "TMenu.h"

///////////////////////////////////////////////////////////
// Constructor:                                          //
// Mas inisialisasiones...                               //
///////////////////////////////////////////////////////////
TMenu::TMenu (int pnumOpciones)
{
   // Inicializamos la ventana
   minX = minY = 0;
   maxX = STDP::TamX() - 1; maxY = STDP::TamY() - 1;

   // Ponemos el numero de opciones
   if (pnumOpciones <= 0) numOpciones = 4;
                     else numOpciones = pnumOpciones;

   // Inicializamos las opciones
   opcion = new TEtiqueta[numOpciones];
   if (opcion == NULL)
   {
      cerr << "Error al reservar memoria en TMenu.\n";
      exit (-1);
   }
   for (int i=0; i<numOpciones; i++)
   {
      char cad[20];
      sprintf (cad, "Opcion %i", i);
      opcion[i].PonCadena    (cad);
      opcion[i].DefineVentana(minX, minY, maxX, maxY);
      opcion[i].Posicionar   ((STDP::TamX()-8)/2, 5 + 2*i);
      opcion[i].PonAtributos (STDP_A_NORMAL, STDP_C_BLANCO, STDP_C_NEGRO);
   }

   // Le ponemos titulo al menu
   titulo.PonCadena    ("Menu de opciones:");
   titulo.DefineVentana(minX, minY,maxX,maxY);
   titulo.Posicionar   ((STDP::TamX()-17)/2, 3);
   titulo.PonAtributos (STDP_A_NEGRITA, STDP_C_ROJO, STDP_C_NEGRO);

   // Se empieza con la primera opcion seleccionada
   seleccion = 0;

   // No hay marco por defecto
   marco.activo = false;
   marco.cadena = NULL;
   marco.FPS = 1;
   marco.Atr=STDP_A_NORMAL; marco.CT=STDP_C_NEGRO; marco.CF=STDP_C_BLANCO;

   // Modo de fondo por defecto
   modo = TMENU_MODO_PLANO;
   color = STDP_C_NEGRO;
}

///////////////////////////////////////////////////////////
// Destructor:                                           //
// Eliminamos la basurilla... xD                         //
///////////////////////////////////////////////////////////
TMenu::~TMenu ()
{
   if (opcion != NULL) { delete [] opcion; opcion = NULL; }
   if (marco.cadena != NULL) { delete [] marco.cadena; marco.cadena = NULL; }
}

///////////////////////////////////////////////////////////
// PonAtrOpcion:                                         //
// Le pone colorines a una opsion...                     //
///////////////////////////////////////////////////////////
void
TMenu::PonAtrOpcion (int numopc, int Atr, int CT, int CF)
{
   if (numopc >= 0 && numopc < numOpciones)
      opcion[numopc].PonAtributos (Atr, CT, CF);
}
///////////////////////////////////////////////////////////
// PonAtrTitulo:                                         //
// Le pone colorines ar titulin...                       //
///////////////////////////////////////////////////////////
void
TMenu::PonAtrTitulo (int Atr, int CT, int CF)
{  titulo.PonAtributos (Atr, CT, CF);  }

///////////////////////////////////////////////////////////
// PonPosOpcion:                                         //
// Posiciona la opcion...                                //
///////////////////////////////////////////////////////////
void
TMenu::PonPosOpcion (int numopc, int x, int y)
{
   if (numopc >= 0 && numopc < numOpciones)
      opcion[numopc].Posicionar (x, y);
}

///////////////////////////////////////////////////////////
// PonPosTitulo:                                         //
// Posiciona er titulin...                               //
///////////////////////////////////////////////////////////
void
TMenu::PonPosTitulo (int x, int y) { titulo.Posicionar (x, y); }

///////////////////////////////////////////////////////////
// PonFondo:                                             //
// Le dise que fondo queremos pal menu...                //
///////////////////////////////////////////////////////////
void
TMenu::PonFondo (int pmodo, int pcolor)
{
   if (STDP::EsColor (pcolor)) color = pcolor;
   if ((modo = pmodo) == TMENU_MODO_PLANO)
   {
      titulo.PonAtributos (titulo.Atributo(), titulo.ColorTexto(), color);
      for (int i=0; i<numOpciones; i++)
         opcion[i].PonAtributos(opcion[i].Atributo(),
                                opcion[i].ColorTexto(), color);
   }
}

///////////////////////////////////////////////////////////
// PonAtributos:                                         //
// Le pone los colores a todas las opciones...           //
///////////////////////////////////////////////////////////
void
TMenu::PonAtributos (int Atr, int CT, int CF)
{ for (int i=0; i<numOpciones; i++) opcion[i].PonAtributos(Atr,CT,CF); }

///////////////////////////////////////////////////////////
// PonCadenaOpcion:                                      //
// Pone el texto de la opcion...                         //
///////////////////////////////////////////////////////////
void
TMenu::PonCadenaOpcion (int numopc, const char *cad)
{
   if (numopc >= 0 && numopc < numOpciones)
   {
      opcion[numopc].PonCadena (cad);
      int x = (maxX-minX-opcion[numopc].Tamanyo())/2;
      opcion[numopc].Posicionar (x, opcion[numopc].PosY());
   }
}

///////////////////////////////////////////////////////////
// PonCadenaTitulo:                                      //
// Pone el texto der titulin...                          //
///////////////////////////////////////////////////////////
void
TMenu::PonCadenaTitulo (const char *cad)
{
   titulo.PonCadena (cad);
   int x = (maxX-minX-titulo.Tamanyo())/2;
   titulo.Posicionar (x, titulo.PosY());
}

///////////////////////////////////////////////////////////
// PonSeleccion:                                         //
// Selecciona una opcion...                              //
///////////////////////////////////////////////////////////
void
TMenu::PonSeleccion (int numopc)
{
   if (numopc >= 0 && numopc < numOpciones)
      seleccion = numopc;
}

///////////////////////////////////////////////////////////
// DefineVentana:                                        //
// Define el tamanyo de la ventana del menu...           //
///////////////////////////////////////////////////////////
void
TMenu::DefineVentana (int pminx, int pminy, int pmaxx, int pmaxy)
{
   if (pminx < pmaxx && pminy < pmaxy && pminx >= 0 && pminy >= 0 &&
       pmaxx < STDP::TamX() && pmaxy < STDP::TamY())
   {
      minX = pminx; minY = pminy; maxX = pmaxx; maxY = pmaxy;
      titulo.DefineVentana (minX, minY, maxX, maxY);
      for (int i=0; i<numOpciones; i++)
         opcion[i].DefineVentana (minX, minY, maxX, maxY);
   }
}

///////////////////////////////////////////////////////////
// PonMarco:                                             //
// Establece el marco que rodea el menu...               //
///////////////////////////////////////////////////////////
void
TMenu::PonMarco (const char *cad, int FPS, int Atr, int CT, int CF)
{
   // Activamos el marco
   marco.activo = true;

   // Copiamos la cadena
   if (marco.cadena != NULL) delete [] marco.cadena;
   if (cad != NULL)
   {
      int lon = strlen (cad);
      marco.cadena = new char[lon + 1];
      if (marco.cadena == NULL)
      {
         cerr << "No se ha podido reservar memoria en TMenu(PonMarco)\n";
         exit (-1);
      }
      strcpy (marco.cadena, cad);
      marco.largo = lon;
   }
   else
   {
      marco.cadena = new char[2];
      marco.cadena[0] = ' ';
      marco.cadena[1] = '\0';
      marco.largo = 1;
   }

   // Y ahora los colores
   if (STDP::EsAtributo (Atr)) marco.Atr = Atr;
   if     (STDP::EsColor (CT)) marco.CT = CT;
   if     (STDP::EsColor (CF)) marco.CF = CF;
   if (FPS > 0) marco.FPS = FPS; else marco.FPS = 1;
}

///////////////////////////////////////////////////////////
// Ejecutar:                                             //
// Ejecuta el menu hasta que se seleccione una opcion... //
///////////////////////////////////////////////////////////
int
TMenu::Ejecutar (double FPS)
{
   bool Fin = false;
   double t;

   do
   {
      // Tomamos el tiempo
      t = clock() + 1.0/FPS * CLOCKS_PER_SEC;

      // Dibujamos
      STDP::BorraPantalla();
      Dibujar();

      // Comprobamos el teclado
      switch (STDP::ObtenTecla())
      {
         case STDP_T_CURSOR_ARRIBA:
         {
            if (--seleccion < 0) seleccion = numOpciones - 1;
            break;
         }
         case STDP_T_CURSOR_ABAJO:
         {
            if (++seleccion >= numOpciones) seleccion = 0;
            break;
         }
         case STDP_T_ENTER:
         {
            Fin = true;
            break;
         }
      }

      // Y esperamos hasta el siguiente fotograma
      while (clock() < t);
   } while (!Fin);

   return seleccion;
}

///////////////////////////////////////////////////////////
// Dibujar:                                              //
// Dibuja el menu de opciones en pantalla...             //
///////////////////////////////////////////////////////////
void
TMenu::Dibujar ()
{
   static int  color = 0;
   static long t = clock();
   static int chnext = 0;

   // Dibujamos el fondo
   switch (modo)
   {
      case TMENU_MODO_PLANO:
      default:
      {
         STDP::CambiaColor (STDP_A_NORMAL, STDP_C_NEGRO, this->color);
         STDP::DibujaCuadrado (minX, minY, maxX, maxY, ' ');
         break;
      }
   }

   // Ahora dibujamos titulo y opciones
   titulo.Dibujar();
   for (int i=0; i<numOpciones; i++)
   {
      if (seleccion == i)
      {
         int atr = opcion[i].Atributo();
         int ct  = opcion[i].ColorTexto();
         int cf  = opcion[i].ColorFondo();
         color = (color + 1)%8;
         opcion[i].PonAtributos (atr, color, cf);
         opcion[i].Dibujar();
         opcion[i].PonAtributos (atr, ct, cf);
      }
      else
         opcion[i].Dibujar();
   }

   // Y ahora dibujamos el marco
   if (marco.activo)
   {
      long nch;
      if ((marco.FPS * (clock() - t) / CLOCKS_PER_SEC))
      {
         if (++chnext >= marco.largo) chnext = 0;
         t = clock();
      }
      nch = chnext;
      STDP::CambiaColor (marco.Atr, marco.CT, marco.CF);
      STDP::PonCursor (minX, minY);
      for (int i=minX; i<=maxX; i++)
      {
         STDP::EscribeCh (marco.cadena[nch]);
         if (++nch >= marco.largo) nch = 0;
      }
      for (int i=minY+1; i<=maxY-1; i++)
      {
         STDP::PonCursor (maxX, i);
         STDP::EscribeCh (marco.cadena[nch]);
         if (++nch >= marco.largo) nch = 0;
      }
      for (int i=maxX; i>=minX; i--)
      {
         STDP::PonCursor (i, maxY);
         STDP::EscribeCh (marco.cadena[nch]);
         if (++nch >= marco.largo) nch = 0;
      }
      for (int i=maxY-1; i>=minY+1; i--)
      {
         STDP::PonCursor (minX, i);
         STDP::EscribeCh (marco.cadena[nch]);
         if (++nch >= marco.largo) nch = 0;
      }
   }
}
