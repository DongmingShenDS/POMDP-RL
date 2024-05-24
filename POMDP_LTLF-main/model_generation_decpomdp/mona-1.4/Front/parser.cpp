/* A Bison parser, made by GNU Bison 3.0.5.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.0.5"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* Copy the first part of user declarations.  */
#line 21 "parser.ypp" /* yacc.c:339  */

#include <stdio.h> 
#include <string.h>
#include "untyped.h"
#include "env.h"

extern MonaUntypedAST *untypedAST;
extern Options options;
extern char *file;
extern bool anyUniverses;

extern void loadFile(char *);
extern void yyerror(const char *);
extern int yylex(); 

#define POS(p) Pos(p.first_line, p.first_column, file)

void check_bits(char *s)
{
  int i;
  for (i = 0; s[i] != '\0'; i++)
    if (s[i] != '0' && s[i] != '1')
      yyerror("parse error");           
}


#line 93 "parser.cpp" /* yacc.c:339  */

# ifndef YY_NULLPTR
#  if defined __cplusplus && 201103L <= __cplusplus
#   define YY_NULLPTR nullptr
#  else
#   define YY_NULLPTR 0
#  endif
# endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* In a future release of Bison, this section will be replaced
   by #include "y.tab.h".  */
#ifndef YY_YY_Y_TAB_H_INCLUDED
# define YY_YY_Y_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    tokALL0 = 258,
    tokALL1 = 259,
    tokALL2 = 260,
    tokAND = 261,
    tokARROW = 262,
    tokASSERT = 263,
    tokBIIMPL = 264,
    tokCOLON = 265,
    tokCOMMA = 266,
    tokCONST = 267,
    tokDEFAULT1 = 268,
    tokDEFAULT2 = 269,
    tokDOT = 270,
    tokEMPTY = 271,
    tokEQUAL = 272,
    tokEX0 = 273,
    tokEX1 = 274,
    tokEX2 = 275,
    tokFALSE = 276,
    tokGREATER = 277,
    tokGREATEREQ = 278,
    tokGUIDE = 279,
    tokIMPL = 280,
    tokIN = 281,
    tokINTER = 282,
    tokINTERVAL = 283,
    tokLBRACE = 284,
    tokLBRACKET = 285,
    tokLESS = 286,
    tokLESSEQ = 287,
    tokLET0 = 288,
    tokLET1 = 289,
    tokLET2 = 290,
    tokLPAREN = 291,
    tokMACRO = 292,
    tokMAX = 293,
    tokMIN = 294,
    tokMINUS = 295,
    tokMODULO = 296,
    tokNOT = 297,
    tokNOTEQUAL = 298,
    tokNOTIN = 299,
    tokOR = 300,
    tokPLUS = 301,
    tokPRED = 302,
    tokRBRACE = 303,
    tokRESTRICT = 304,
    tokRBRACKET = 305,
    tokUNIVROOT = 306,
    tokRPAREN = 307,
    tokSEMICOLON = 308,
    tokSETMINUS = 309,
    tokSLASH = 310,
    tokSTAR = 311,
    tokVERIFY = 312,
    tokSUB = 313,
    tokTREE = 314,
    tokTRUE = 315,
    tokUNION = 316,
    tokUNIVERSE = 317,
    tokUP = 318,
    tokVAR0 = 319,
    tokVAR1 = 320,
    tokVAR2 = 321,
    tokWHERE = 322,
    tokINCLUDE = 323,
    tokIMPORT = 324,
    tokEXPORT = 325,
    tokPREFIX = 326,
    tokM2LSTR = 327,
    tokM2LTREE = 328,
    tokLASTPOS = 329,
    tokINSTATESPACE = 330,
    tokEXECUTE = 331,
    tokTYPE = 332,
    tokSOMETYPE = 333,
    tokVARIANT = 334,
    tokSUCC = 335,
    tokWS1S = 336,
    tokWS2S = 337,
    tokTREEROOT = 338,
    tokCONSTTREE = 339,
    tokALLPOS = 340,
    tokINT = 341,
    tokNAME = 342,
    tokSTRING = 343,
    LOW = 344
  };
#endif
/* Tokens.  */
#define tokALL0 258
#define tokALL1 259
#define tokALL2 260
#define tokAND 261
#define tokARROW 262
#define tokASSERT 263
#define tokBIIMPL 264
#define tokCOLON 265
#define tokCOMMA 266
#define tokCONST 267
#define tokDEFAULT1 268
#define tokDEFAULT2 269
#define tokDOT 270
#define tokEMPTY 271
#define tokEQUAL 272
#define tokEX0 273
#define tokEX1 274
#define tokEX2 275
#define tokFALSE 276
#define tokGREATER 277
#define tokGREATEREQ 278
#define tokGUIDE 279
#define tokIMPL 280
#define tokIN 281
#define tokINTER 282
#define tokINTERVAL 283
#define tokLBRACE 284
#define tokLBRACKET 285
#define tokLESS 286
#define tokLESSEQ 287
#define tokLET0 288
#define tokLET1 289
#define tokLET2 290
#define tokLPAREN 291
#define tokMACRO 292
#define tokMAX 293
#define tokMIN 294
#define tokMINUS 295
#define tokMODULO 296
#define tokNOT 297
#define tokNOTEQUAL 298
#define tokNOTIN 299
#define tokOR 300
#define tokPLUS 301
#define tokPRED 302
#define tokRBRACE 303
#define tokRESTRICT 304
#define tokRBRACKET 305
#define tokUNIVROOT 306
#define tokRPAREN 307
#define tokSEMICOLON 308
#define tokSETMINUS 309
#define tokSLASH 310
#define tokSTAR 311
#define tokVERIFY 312
#define tokSUB 313
#define tokTREE 314
#define tokTRUE 315
#define tokUNION 316
#define tokUNIVERSE 317
#define tokUP 318
#define tokVAR0 319
#define tokVAR1 320
#define tokVAR2 321
#define tokWHERE 322
#define tokINCLUDE 323
#define tokIMPORT 324
#define tokEXPORT 325
#define tokPREFIX 326
#define tokM2LSTR 327
#define tokM2LTREE 328
#define tokLASTPOS 329
#define tokINSTATESPACE 330
#define tokEXECUTE 331
#define tokTYPE 332
#define tokSOMETYPE 333
#define tokVARIANT 334
#define tokSUCC 335
#define tokWS1S 336
#define tokWS2S 337
#define tokTREEROOT 338
#define tokCONSTTREE 339
#define tokALLPOS 340
#define tokINT 341
#define tokNAME 342
#define tokSTRING 343
#define LOW 344

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 51 "parser.ypp" /* yacc.c:355  */
 
  char *string;
  int integer;
  MonaUntypedAST *monaUntypedAST;
  DeclarationList *declarationList;
  Declaration *declaration;
  UntypedExp *untypedExp;
  ArithExp *arithExp;
  ParDeclList *parDeclList;
  BindExpList *bindExpList;
  BindExp *bindExp;
  UntypedExpList *untypedExpList;
  NameList *nameList;
  Name *name;
  VarDeclList *varDeclList;
  GuideFuncList *guideFuncList;
  GuideFunc *guideFunc;
  UnivList *univList;
  Univ *univ;
  ImportMapList *mapList;
  VariantList *variantList;
  ComponentList *componentList;
  ConstNode *constNode;
  ConstNodeList *constNodeList;

#line 337 "parser.cpp" /* yacc.c:355  */
};

typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif

/* Location type.  */
#if ! defined YYLTYPE && ! defined YYLTYPE_IS_DECLARED
typedef struct YYLTYPE YYLTYPE;
struct YYLTYPE
{
  int first_line;
  int first_column;
  int last_line;
  int last_column;
};
# define YYLTYPE_IS_DECLARED 1
# define YYLTYPE_IS_TRIVIAL 1
#endif


extern YYSTYPE yylval;
extern YYLTYPE yylloc;
int yyparse (void);

#endif /* !YY_YY_Y_TAB_H_INCLUDED  */

/* Copy the second part of user declarations.  */

#line 368 "parser.cpp" /* yacc.c:358  */

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif

#ifndef YY_ATTRIBUTE
# if (defined __GNUC__                                               \
      && (2 < __GNUC__ || (__GNUC__ == 2 && 96 <= __GNUC_MINOR__)))  \
     || defined __SUNPRO_C && 0x5110 <= __SUNPRO_C
#  define YY_ATTRIBUTE(Spec) __attribute__(Spec)
# else
#  define YY_ATTRIBUTE(Spec) /* empty */
# endif
#endif

#ifndef YY_ATTRIBUTE_PURE
# define YY_ATTRIBUTE_PURE   YY_ATTRIBUTE ((__pure__))
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# define YY_ATTRIBUTE_UNUSED YY_ATTRIBUTE ((__unused__))
#endif

#if !defined _Noreturn \
     && (!defined __STDC_VERSION__ || __STDC_VERSION__ < 201112)
# if defined _MSC_VER && 1200 <= _MSC_VER
#  define _Noreturn __declspec (noreturn)
# else
#  define _Noreturn YY_ATTRIBUTE ((__noreturn__))
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN \
    _Pragma ("GCC diagnostic push") \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")\
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif


#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL \
             && defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
  YYLTYPE yyls_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE) + sizeof (YYLTYPE)) \
      + 2 * YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYSIZE_T yynewbytes;                                            \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / sizeof (*yyptr);                          \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, (Count) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYSIZE_T yyi;                         \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  11
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   2158

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  90
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  29
/* YYNRULES -- Number of rules.  */
#define YYNRULES  157
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  414

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   344

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   147,   147,   151,   153,   155,   157,   160,   163,   165,
     170,   172,   174,   177,   179,   181,   183,   185,   187,   189,
     191,   193,   196,   199,   201,   204,   207,   209,   211,   213,
     215,   217,   219,   223,   225,   227,   229,   231,   233,   235,
     237,   239,   241,   243,   245,   247,   249,   251,   253,   255,
     257,   259,   261,   263,   266,   268,   270,   272,   274,   276,
     278,   280,   282,   284,   286,   288,   290,   292,   295,   297,
     299,   301,   303,   305,   307,   309,   311,   313,   315,   317,
     319,   321,   323,   325,   327,   329,   332,   335,   337,   339,
     341,   344,   346,   350,   352,   354,   356,   358,   361,   363,
     365,   369,   372,   375,   378,   381,   384,   387,   390,   393,
     396,   401,   404,   409,   413,   416,   419,   422,   425,   428,
     433,   436,   439,   442,   447,   450,   453,   457,   460,   465,
     468,   473,   476,   481,   485,   488,   493,   496,   498,   502,
     505,   508,   512,   515,   518,   521,   524,   527,   530,   535,
     538,   543,   545,   547,   551,   553,   558,   561
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "tokALL0", "tokALL1", "tokALL2",
  "tokAND", "tokARROW", "tokASSERT", "tokBIIMPL", "tokCOLON", "tokCOMMA",
  "tokCONST", "tokDEFAULT1", "tokDEFAULT2", "tokDOT", "tokEMPTY",
  "tokEQUAL", "tokEX0", "tokEX1", "tokEX2", "tokFALSE", "tokGREATER",
  "tokGREATEREQ", "tokGUIDE", "tokIMPL", "tokIN", "tokINTER",
  "tokINTERVAL", "tokLBRACE", "tokLBRACKET", "tokLESS", "tokLESSEQ",
  "tokLET0", "tokLET1", "tokLET2", "tokLPAREN", "tokMACRO", "tokMAX",
  "tokMIN", "tokMINUS", "tokMODULO", "tokNOT", "tokNOTEQUAL", "tokNOTIN",
  "tokOR", "tokPLUS", "tokPRED", "tokRBRACE", "tokRESTRICT", "tokRBRACKET",
  "tokUNIVROOT", "tokRPAREN", "tokSEMICOLON", "tokSETMINUS", "tokSLASH",
  "tokSTAR", "tokVERIFY", "tokSUB", "tokTREE", "tokTRUE", "tokUNION",
  "tokUNIVERSE", "tokUP", "tokVAR0", "tokVAR1", "tokVAR2", "tokWHERE",
  "tokINCLUDE", "tokIMPORT", "tokEXPORT", "tokPREFIX", "tokM2LSTR",
  "tokM2LTREE", "tokLASTPOS", "tokINSTATESPACE", "tokEXECUTE", "tokTYPE",
  "tokSOMETYPE", "tokVARIANT", "tokSUCC", "tokWS1S", "tokWS2S",
  "tokTREEROOT", "tokCONSTTREE", "tokALLPOS", "tokINT", "tokNAME",
  "tokSTRING", "LOW", "$accept", "start", "header", "declarations",
  "declaration", "exp", "arith_exp", "par_list", "defs", "def", "set_body",
  "non_empty_set_body", "exp_list", "non_empty_exp_list", "universe",
  "name", "name_list", "name_where_list", "func_list", "func", "univs",
  "univ", "where", "map_list", "variant_list", "component_list",
  "constnode", "constnode_list", "optstring", YY_NULLPTR
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344
};
# endif

#define YYPACT_NINF -324

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-324)))

#define YYTABLE_NINF -1

#define yytable_value_is_error(Yytable_value) \
  (!!((Yytable_value) == (-1)))

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
       2,   -28,   -10,    -3,    27,    30,   581,  -324,  -324,  -324,
    -324,  -324,   -40,    55,    55,   666,   -40,    21,    51,    57,
     -40,    55,    55,  -324,   -40,   465,   -40,   -40,   -40,   666,
     -40,   666,   666,   666,   -40,    61,    63,    23,    12,  -324,
     -40,   -40,    55,    55,    28,    68,    86,    94,   -40,   104,
     666,   -34,   111,   116,   130,   132,   133,   -40,  -324,  -324,
    -324,   581,   742,   136,    34,   172,   -40,   -40,   -40,   148,
     150,   784,   170,   -40,   -40,   666,   180,   -40,   -40,   187,
     146,   189,   192,   835,   157,  -324,   183,   205,   201,   194,
     199,   877,     4,   158,   158,  2053,    16,   666,   666,  -324,
     666,   666,   -40,   218,   176,   219,   181,   -40,   -40,   182,
     145,   151,   666,   185,   666,   919,   666,   224,   666,   666,
     666,   666,   666,   191,  -324,   666,   666,   156,   666,   666,
     666,   666,   666,   666,   666,   666,   -27,   666,   666,   666,
     -27,  -324,   666,   -27,   -27,   666,   666,  -324,   666,   666,
     234,   666,   236,   202,   243,   247,  -324,   -27,   206,   208,
     961,   666,   252,   255,   230,  -324,   -40,   465,   465,  -324,
     666,   -40,   666,   666,   666,  -324,   666,    91,   666,    96,
    1003,  1045,   -17,  1087,  1129,   214,   -23,  -324,   -40,  -324,
     217,   221,  -324,   260,   264,  1171,  -324,  1213,  -324,  1255,
     -40,  1297,  1339,  1381,  1423,  1465,  -324,  2053,  1969,  -324,
     200,   200,   200,   419,  2095,    79,   200,   200,   -27,   -27,
    -324,    62,  -324,   200,  2095,    83,    95,   254,  -324,  -324,
    2095,   249,  1507,   225,  -324,  1969,   -40,  1969,   -40,  -324,
     666,   666,   155,   262,   265,  -324,  1969,   666,   666,   -40,
    -324,  -324,  -324,  1969,  -324,  2011,  1969,  1969,  1549,   267,
     -40,   -40,   -40,   -40,   229,    34,  1591,   273,   239,  -324,
      55,  -324,  -324,  -324,  -324,  -324,  -324,  -324,  -324,  -324,
     -40,   240,   666,  -324,   -40,   -40,    13,   245,  -324,   666,
     -40,  -324,   -40,    36,    65,   -27,   666,   -27,   -27,   -27,
     666,   666,  -324,  -324,  -324,  1969,  1969,  -324,   666,   666,
    1969,  1969,   285,  -324,   666,   288,   290,    34,    34,   276,
     291,  -324,   666,   289,   256,   300,  -324,  1633,   268,   269,
     -40,    -8,  -324,  1675,   305,   312,  -324,    65,     8,    65,
    -324,  -324,     8,  -324,  1717,  1759,   -40,  1801,   115,   115,
     313,   314,   666,   115,  1843,   666,  -324,   -40,  -324,  -324,
    -324,  -324,   315,   317,   271,   -40,   -40,   -40,  -324,  -324,
     277,  -324,  -324,  -324,   115,   115,  1885,  -324,  -324,  1927,
     260,   -40,   -40,   319,   320,   322,   292,   283,  -324,  -324,
    -324,  -324,  -324,  -324,  -324,   327,   -40,   -40,   -40,    -6,
    -324,   -40,  -324,   287,   294,  -324,   329,   295,  -324,  -324,
    -324,   -40,  -324,  -324
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       7,     0,     0,     0,     0,     0,     0,     5,     6,     3,
       4,     1,     0,   125,   125,     0,     0,     0,     0,    76,
       0,   125,   125,    66,     0,   115,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    52,   157,   125,    65,
       0,     0,   125,   125,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    75,   126,
       2,     9,     0,    33,   140,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   132,   119,   118,     0,   114,     0,   112,     0,     0,
       0,     0,     0,    39,    38,    50,     0,     0,     0,   156,
       0,     0,     0,   138,     0,   135,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     8,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    26,     0,     0,     0,     0,     0,    54,   121,     0,
     130,     0,   128,     0,     0,     0,    10,     0,     0,     0,
       0,     0,     0,     0,     0,    11,     0,     0,     0,    77,
       0,     0,     0,     0,     0,    34,     0,     0,     0,     0,
       0,     0,    33,     0,     0,     0,     0,    12,     0,    16,
       0,     0,    29,   142,     0,     0,    30,     0,    28,     0,
       0,     0,     0,     0,     0,     0,    31,    48,    47,    53,
      44,    43,    42,    46,    36,    79,    40,    41,     0,     0,
      98,    72,    99,    45,    37,    49,    71,    80,    74,    73,
      35,    78,   123,     0,   120,   139,     0,    58,     0,   124,
       0,     0,     0,     0,     0,    68,    55,     0,     0,     0,
     131,   117,   116,    61,   111,   113,    62,    63,     0,     0,
       0,     0,     0,     0,     0,   140,     0,     0,     0,    92,
     125,    51,    27,    87,    19,   136,   137,   134,    17,    18,
       0,     0,     0,    83,     0,     0,   148,     0,    89,     0,
       0,    91,     0,     0,    97,     0,     0,     0,     0,     0,
       0,     0,    64,   129,   127,    59,    60,    15,     0,     0,
      56,    57,     0,    24,     0,   109,   106,   140,   140,     0,
     110,    21,     0,     0,     0,     0,    81,     0,     0,     0,
       0,     0,    32,     0,     0,     0,   100,    94,    70,    93,
      96,    95,    69,   122,     0,     0,     0,     0,     0,     0,
     107,   108,     0,     0,     0,     0,    67,     0,    82,    84,
      88,   145,   147,     0,     0,     0,     0,     0,    13,    14,
       0,    25,   104,   101,     0,     0,     0,   105,    22,     0,
     142,     0,     0,   146,     0,     0,   153,     0,   133,   102,
     103,    23,    20,   141,   144,   150,     0,     0,     0,     0,
      90,     0,   143,     0,     0,   152,   155,     0,   149,    85,
      86,     0,   151,   154
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -324,  -324,  -324,   281,  -324,   282,  -106,  -178,   -22,  -324,
    -324,    -4,  -324,    43,   -11,   -12,  -212,     0,   184,  -324,
     160,  -324,  -248,   -31,  -323,   -50,   -15,   -55,  -324
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     5,     6,    60,    61,    83,   221,   264,    86,    87,
      84,    85,   233,   234,    67,    63,   153,    65,    80,    81,
     104,   105,   150,   281,   287,   364,   406,   407,   100
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      64,   268,   116,    68,    72,    89,    90,   361,    64,   218,
      77,    78,    79,   219,    88,    88,    88,   320,    92,   148,
      76,   176,    96,   127,   330,     7,   304,   102,   103,    64,
      11,   107,   108,   178,   226,   271,   113,   228,   229,   117,
     177,   106,    66,     8,   362,   123,   405,    59,   101,   331,
       9,   242,   179,    59,   152,    64,    64,    73,   394,   220,
      59,   158,   159,   275,    59,    64,    64,   154,   155,   350,
     351,   147,   328,   402,     1,     2,   295,   162,   163,    59,
      10,    59,   297,     3,     4,    66,   182,    74,   336,   125,
      64,   298,   299,    75,   127,    64,    64,    97,   127,    98,
     128,   149,   185,   296,   110,   129,   130,   190,   191,   132,
     133,    99,   293,   294,   134,   135,   109,   298,   299,   136,
     298,   299,   111,   136,   222,   140,   137,   138,   222,   140,
     112,   222,   222,   142,   143,   144,   300,   142,   143,   144,
     114,   145,   147,   259,   146,   222,   147,   118,   267,   254,
     298,   299,   119,   260,    79,   261,   262,   263,   260,    88,
     261,   262,   263,   251,   252,   265,   120,   265,   121,   122,
     372,   373,   148,   127,   276,   377,   103,   260,    59,   261,
     262,   263,   151,    59,   101,   133,   116,   157,   286,   337,
     161,   339,   340,   341,   164,   295,   389,   390,   136,   165,
     166,   297,    59,   167,   140,   169,   222,   222,   307,   170,
     298,   299,   142,   143,   144,   127,   171,    -1,   172,   146,
     173,   147,    -1,    -1,    64,   174,   152,   133,   186,   187,
     188,    -1,    -1,   193,   189,   192,   303,   312,   196,   194,
     136,   200,   209,    -1,   206,   236,   140,   238,   315,   316,
     317,   318,   239,   240,   142,   143,   144,   241,   243,   324,
     244,   146,   247,   147,   127,   248,   249,   274,   325,   127,
     278,   280,   152,   329,   279,   282,   133,   302,   334,   308,
     335,   319,   309,   222,   314,   222,   222,   222,    62,   136,
     322,   323,   326,   352,   136,   140,   346,    71,   332,   348,
     140,   349,   353,   142,   143,   144,   355,   357,   356,   143,
     144,    91,   147,    93,    94,    95,   366,   147,   286,   363,
     359,   360,   367,   383,   374,   375,   381,   382,   399,   388,
     396,   397,   115,   398,   370,   400,   265,   265,   401,   409,
     411,   265,   124,    62,   343,   380,   410,   412,   277,   393,
     250,   408,   387,   384,   385,   386,   413,   160,     0,     0,
       0,     0,   265,   265,     0,     0,     0,     0,     0,   286,
     395,     0,     0,     0,     0,     0,     0,     0,     0,   180,
     181,     0,   183,   184,   286,   403,   404,   386,     0,   363,
       0,     0,     0,     0,   195,     0,   197,     0,   199,   386,
     201,   202,   203,   204,   205,     0,     0,   207,   208,     0,
     210,   211,   212,   213,   214,   215,   216,   217,     0,   223,
     224,   225,     0,     0,   227,   125,     0,   230,   231,     0,
     232,   235,     0,   237,   127,     0,   128,     0,     0,     0,
       0,   129,   130,   246,   131,   132,   133,     0,     0,     0,
     134,   135,   253,     0,   255,   256,   257,     0,   258,   136,
     266,     0,   137,   138,   139,   140,     0,     0,    12,    13,
      14,     0,     0,   142,   143,   144,     0,   145,     0,     0,
     146,    19,   147,    20,    21,    22,    23,     0,     0,     0,
       0,     0,     0,    82,    25,     0,     0,     0,    26,    27,
      28,    29,     0,    31,    32,     0,     0,    33,     0,     0,
       0,     0,     0,     0,    35,     0,    36,     0,     0,     0,
       0,     0,   305,   306,    69,    39,     0,     0,     0,   310,
     311,     0,     0,     0,    45,    46,    47,     0,     0,     0,
      49,     0,    70,    52,    53,    54,     0,     0,    55,    56,
       0,    58,    59,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   327,     0,     0,     0,     0,     0,
       0,   333,     0,     0,     0,     0,     0,     0,   338,     0,
       0,     0,   342,   232,    12,    13,    14,     0,     0,    15,
     344,   345,     0,    16,    17,    18,   347,    19,     0,    20,
      21,    22,    23,     0,   354,    24,     0,     0,     0,     0,
      25,     0,     0,     0,    26,    27,    28,    29,    30,    31,
      32,     0,     0,    33,     0,     0,     0,     0,    34,     0,
      35,     0,    36,     0,   376,     0,     0,   379,    37,     0,
      38,    39,     0,    40,     0,    41,    42,    43,     0,    44,
      45,    46,    47,     0,     0,    48,    49,    50,    51,    52,
      53,    54,     0,     0,    55,    56,    57,    58,    59,    12,
      13,    14,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    19,     0,    20,    21,    22,    23,     0,     0,
       0,     0,     0,     0,     0,    25,     0,     0,     0,    26,
      27,    28,    29,     0,    31,    32,     0,     0,    33,     0,
       0,     0,     0,     0,     0,    35,     0,    36,     0,     0,
       0,     0,     0,     0,     0,    69,    39,     0,     0,     0,
       0,     0,     0,     0,     0,    45,    46,    47,     0,     0,
       0,    49,     0,    70,    52,    53,    54,     0,   125,    55,
      56,   126,    58,    59,     0,     0,     0,   127,     0,   128,
       0,     0,     0,     0,   129,   130,     0,   131,   132,   133,
       0,     0,     0,   134,   135,     0,     0,     0,     0,     0,
       0,     0,   136,     0,     0,   137,   138,   139,   140,     0,
     125,     0,     0,   126,     0,   141,   142,   143,   144,   127,
     145,   128,     0,   146,     0,   147,   129,   130,     0,   131,
     132,   133,     0,     0,     0,   134,   135,     0,     0,     0,
       0,     0,     0,     0,   136,     0,     0,   137,   138,   139,
     140,     0,     0,     0,     0,     0,     0,   156,   142,   143,
     144,   125,   145,     0,   126,   146,   168,   147,     0,     0,
     127,     0,   128,     0,     0,     0,     0,   129,   130,     0,
     131,   132,   133,     0,     0,     0,   134,   135,     0,     0,
       0,     0,     0,     0,     0,   136,     0,     0,   137,   138,
     139,   140,     0,   125,     0,     0,   126,     0,     0,   142,
     143,   144,   127,   145,   128,     0,   146,     0,   147,   129,
     130,     0,   131,   132,   133,     0,     0,     0,   134,   135,
       0,     0,     0,     0,     0,     0,     0,   136,     0,     0,
     137,   138,   139,   140,     0,   125,     0,     0,   126,   175,
       0,   142,   143,   144,   127,   145,   128,     0,   146,     0,
     147,   129,   130,     0,   131,   132,   133,     0,     0,     0,
     134,   135,     0,     0,     0,     0,     0,     0,     0,   136,
       0,     0,   137,   138,   139,   140,     0,   125,     0,     0,
     126,     0,   198,   142,   143,   144,   127,   145,   128,     0,
     146,     0,   147,   129,   130,     0,   131,   132,   133,     0,
       0,     0,   134,   135,     0,     0,     0,     0,     0,     0,
       0,   136,     0,     0,   137,   138,   139,   140,     0,   125,
       0,     0,   126,   245,     0,   142,   143,   144,   127,   145,
     128,     0,   146,     0,   147,   129,   130,     0,   131,   132,
     133,     0,     0,     0,   134,   135,     0,     0,     0,     0,
       0,     0,     0,   136,     0,     0,   137,   138,   139,   140,
       0,   125,     0,     0,   126,   269,   270,   142,   143,   144,
     127,   145,   128,     0,   146,     0,   147,   129,   130,     0,
     131,   132,   133,     0,     0,     0,   134,   135,     0,     0,
       0,     0,     0,     0,     0,   136,     0,     0,   137,   138,
     139,   140,     0,   125,     0,     0,   126,     0,     0,   142,
     143,   144,   127,   145,   128,     0,   146,     0,   147,   129,
     130,     0,   131,   132,   133,     0,     0,     0,   134,   135,
       0,     0,     0,     0,     0,     0,     0,   136,     0,     0,
     137,   138,   139,   140,     0,   125,     0,     0,   126,     0,
     272,   142,   143,   144,   127,   145,   128,     0,   146,     0,
     147,   129,   130,     0,   131,   132,   133,     0,     0,     0,
     134,   135,     0,     0,     0,     0,     0,     0,     0,   136,
       0,     0,   137,   138,   139,   140,     0,   125,     0,     0,
     126,   273,     0,   142,   143,   144,   127,   145,   128,     0,
     146,     0,   147,   129,   130,     0,   131,   132,   133,     0,
       0,     0,   134,   135,     0,     0,     0,     0,     0,     0,
       0,   136,     0,     0,   137,   138,   139,   140,     0,   125,
       0,     0,   126,   283,   284,   142,   143,   144,   127,   145,
     128,     0,   146,     0,   147,   129,   130,     0,   131,   132,
     133,     0,     0,     0,   134,   135,     0,     0,     0,     0,
       0,     0,     0,   136,     0,     0,   137,   138,   139,   140,
       0,   125,     0,     0,   126,     0,   285,   142,   143,   144,
     127,   145,   128,     0,   146,     0,   147,   129,   130,     0,
     131,   132,   133,     0,     0,     0,   134,   135,     0,     0,
       0,     0,     0,     0,     0,   136,     0,     0,   137,   138,
     139,   140,     0,   125,     0,     0,   126,     0,     0,   142,
     143,   144,   127,   145,   128,     0,   146,     0,   147,   129,
     130,     0,   131,   132,   133,     0,     0,     0,   134,   135,
       0,     0,     0,     0,     0,     0,     0,   136,     0,     0,
     137,   138,   139,   140,     0,   125,     0,     0,   126,   288,
     289,   142,   143,   144,   127,   145,   128,     0,   146,     0,
     147,   129,   130,     0,   131,   132,   133,     0,     0,     0,
     134,   135,     0,     0,     0,     0,     0,     0,     0,   136,
       0,     0,   137,   138,   139,   140,     0,   125,     0,     0,
     126,     0,   290,   142,   143,   144,   127,   145,   128,     0,
     146,     0,   147,   129,   130,     0,   131,   132,   133,     0,
       0,     0,   134,   135,     0,     0,     0,     0,     0,     0,
       0,   136,     0,     0,   137,   138,   139,   140,     0,   125,
       0,     0,   126,     0,     0,   142,   143,   144,   127,   145,
     128,     0,   146,     0,   147,   129,   130,     0,   131,   132,
     133,     0,     0,     0,   134,   135,     0,     0,     0,     0,
       0,     0,     0,   136,     0,     0,   137,   138,   139,   140,
       0,   125,     0,     0,   126,   291,   292,   142,   143,   144,
     127,   145,   128,     0,   146,     0,   147,   129,   130,     0,
     131,   132,   133,     0,     0,     0,   134,   135,     0,     0,
       0,     0,     0,     0,     0,   136,     0,     0,   137,   138,
     139,   140,     0,   125,     0,     0,   126,     0,   301,   142,
     143,   144,   127,   145,   128,     0,   146,     0,   147,   129,
     130,     0,   131,   132,   133,     0,     0,     0,   134,   135,
       0,     0,     0,     0,     0,     0,     0,   136,     0,     0,
     137,   138,   139,   140,     0,   125,     0,     0,   126,     0,
       0,   142,   143,   144,   127,   145,   128,     0,   146,     0,
     147,   129,   130,     0,   131,   132,   133,     0,     0,     0,
     134,   135,     0,     0,     0,     0,     0,     0,     0,   136,
       0,     0,   137,   138,   139,   140,     0,   125,     0,     0,
     126,     0,   313,   142,   143,   144,   127,   145,   128,     0,
     146,     0,   147,   129,   130,     0,   131,   132,   133,     0,
       0,     0,   134,   135,     0,     0,     0,     0,     0,     0,
       0,   136,     0,     0,   137,   138,   139,   140,     0,   125,
       0,     0,   126,     0,   321,   142,   143,   144,   127,   145,
     128,     0,   146,     0,   147,   129,   130,     0,   131,   132,
     133,     0,     0,     0,   134,   135,     0,     0,     0,     0,
       0,     0,     0,   136,     0,     0,   137,   138,   139,   140,
       0,   125,     0,     0,   126,   358,   365,   142,   143,   144,
     127,   145,   128,     0,   146,     0,   147,   129,   130,     0,
     131,   132,   133,     0,     0,     0,   134,   135,     0,     0,
       0,     0,     0,     0,     0,   136,     0,     0,   137,   138,
     139,   140,     0,   125,     0,     0,   126,     0,     0,   142,
     143,   144,   127,   145,   128,     0,   146,     0,   147,   129,
     130,     0,   131,   132,   133,     0,     0,     0,   134,   135,
       0,     0,     0,     0,     0,     0,     0,   136,     0,     0,
     137,   138,   139,   140,     0,   125,     0,     0,   126,     0,
     368,   142,   143,   144,   127,   145,   128,     0,   146,     0,
     147,   129,   130,     0,   131,   132,   133,     0,     0,     0,
     134,   135,     0,     0,     0,     0,     0,     0,     0,   136,
       0,     0,   137,   138,   139,   140,     0,   125,     0,     0,
     126,     0,   369,   142,   143,   144,   127,   145,   128,     0,
     146,     0,   147,   129,   130,     0,   131,   132,   133,     0,
       0,     0,   134,   135,     0,     0,     0,     0,     0,     0,
       0,   136,     0,     0,   137,   138,   139,   140,     0,   125,
       0,     0,   126,     0,   371,   142,   143,   144,   127,   145,
     128,     0,   146,     0,   147,   129,   130,     0,   131,   132,
     133,     0,     0,     0,   134,   135,     0,     0,     0,     0,
       0,     0,     0,   136,     0,     0,   137,   138,   139,   140,
       0,   125,     0,     0,   126,     0,   378,   142,   143,   144,
     127,   145,   128,     0,   146,     0,   147,   129,   130,     0,
     131,   132,   133,     0,     0,     0,   134,   135,     0,     0,
       0,     0,     0,     0,     0,   136,     0,     0,   137,   138,
     139,   140,     0,   125,     0,     0,   126,     0,   391,   142,
     143,   144,   127,   145,   128,     0,   146,     0,   147,   129,
     130,     0,   131,   132,   133,     0,     0,     0,   134,   135,
       0,     0,     0,     0,     0,     0,     0,   136,     0,     0,
     137,   138,   139,   140,     0,   125,     0,     0,   126,     0,
     392,   142,   143,   144,   127,   145,   128,     0,   146,     0,
     147,   129,   130,     0,   131,   132,   133,     0,     0,     0,
     134,   135,     0,     0,     0,     0,     0,     0,     0,   136,
       0,     0,   137,   138,   139,   140,     0,   125,     0,     0,
     126,     0,     0,   142,   143,   144,   127,   145,   128,     0,
     146,     0,   147,   129,   130,     0,   131,     0,   133,     0,
       0,     0,   134,   135,     0,     0,     0,     0,     0,     0,
       0,   136,     0,     0,   137,   138,   139,   140,     0,     0,
       0,     0,     0,     0,     0,   142,   143,   144,   127,   145,
     128,     0,   146,     0,   147,   129,   130,     0,     0,   132,
     133,     0,     0,     0,   134,   135,     0,     0,     0,     0,
       0,     0,     0,   136,     0,     0,   137,   138,     0,   140,
       0,     0,     0,     0,     0,     0,     0,   142,   143,   144,
     127,   145,   128,     0,   146,     0,   147,   129,   130,     0,
       0,    -1,   133,     0,     0,     0,   134,   135,     0,     0,
       0,     0,     0,     0,     0,   136,     0,     0,   137,    -1,
       0,   140,     0,     0,     0,     0,     0,     0,     0,   142,
     143,   144,     0,    -1,     0,     0,   146,     0,   147
};

static const yytype_int16 yycheck[] =
{
      12,   179,    36,    14,    16,    27,    28,   330,    20,    36,
      21,    22,    24,    40,    26,    27,    28,   265,    30,    36,
      20,    17,    34,    15,    11,    53,   238,    38,    40,    41,
       0,    42,    43,    17,   140,    52,    48,   143,   144,    51,
      36,    41,    30,    53,    52,    57,    52,    87,    36,    36,
      53,   157,    36,    87,    66,    67,    68,    36,   381,    86,
      87,    73,    74,    86,    87,    77,    78,    67,    68,   317,
     318,    63,   284,   396,    72,    73,    40,    77,    78,    87,
      53,    87,    46,    81,    82,    30,    98,    36,    52,     6,
     102,    55,    56,    36,    15,   107,   108,    36,    15,    36,
      17,    67,   102,    41,    36,    22,    23,   107,   108,    26,
      27,    88,   218,   219,    31,    32,    88,    55,    56,    40,
      55,    56,    36,    40,   136,    46,    43,    44,   140,    46,
      36,   143,   144,    54,    55,    56,    41,    54,    55,    56,
      36,    58,    63,    52,    61,   157,    63,    36,    52,   171,
      55,    56,    36,    62,   166,    64,    65,    66,    62,   171,
      64,    65,    66,   167,   168,   177,    36,   179,    36,    36,
     348,   349,    36,    15,   186,   353,   188,    62,    87,    64,
      65,    66,    10,    87,    36,    27,    36,    17,   200,   295,
      10,   297,   298,   299,     7,    40,   374,   375,    40,    53,
      11,    46,    87,    11,    46,    48,   218,   219,    53,    26,
      55,    56,    54,    55,    56,    15,    11,    17,    17,    61,
      26,    63,    22,    23,   236,    26,   238,    27,    10,    53,
      11,    31,    32,    88,    53,    53,   236,   249,    53,    88,
      40,    17,    86,    43,    53,    11,    46,    11,   260,   261,
     262,   263,    50,    10,    54,    55,    56,    10,    52,   270,
      52,    61,    10,    63,    15,    10,    36,    53,   280,    15,
      53,    11,   284,   285,    53,    11,    27,    52,   290,    17,
     292,    52,    17,   295,    17,   297,   298,   299,     6,    40,
      17,    52,    52,    17,    40,    46,    11,    15,    53,    11,
      46,    11,    11,    54,    55,    56,    17,     7,    52,    55,
      56,    29,    63,    31,    32,    33,    11,    63,   330,   331,
      52,    52,    10,    52,    11,    11,    11,    10,    36,    52,
      11,    11,    50,    11,   346,    52,   348,   349,    11,    52,
      11,   353,    61,    61,   301,   357,    52,    52,   188,   380,
     166,   401,   367,   365,   366,   367,   411,    75,    -1,    -1,
      -1,    -1,   374,   375,    -1,    -1,    -1,    -1,    -1,   381,
     382,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    97,
      98,    -1,   100,   101,   396,   397,   398,   399,    -1,   401,
      -1,    -1,    -1,    -1,   112,    -1,   114,    -1,   116,   411,
     118,   119,   120,   121,   122,    -1,    -1,   125,   126,    -1,
     128,   129,   130,   131,   132,   133,   134,   135,    -1,   137,
     138,   139,    -1,    -1,   142,     6,    -1,   145,   146,    -1,
     148,   149,    -1,   151,    15,    -1,    17,    -1,    -1,    -1,
      -1,    22,    23,   161,    25,    26,    27,    -1,    -1,    -1,
      31,    32,   170,    -1,   172,   173,   174,    -1,   176,    40,
     178,    -1,    43,    44,    45,    46,    -1,    -1,     3,     4,
       5,    -1,    -1,    54,    55,    56,    -1,    58,    -1,    -1,
      61,    16,    63,    18,    19,    20,    21,    -1,    -1,    -1,
      -1,    -1,    -1,    28,    29,    -1,    -1,    -1,    33,    34,
      35,    36,    -1,    38,    39,    -1,    -1,    42,    -1,    -1,
      -1,    -1,    -1,    -1,    49,    -1,    51,    -1,    -1,    -1,
      -1,    -1,   240,   241,    59,    60,    -1,    -1,    -1,   247,
     248,    -1,    -1,    -1,    69,    70,    71,    -1,    -1,    -1,
      75,    -1,    77,    78,    79,    80,    -1,    -1,    83,    84,
      -1,    86,    87,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   282,    -1,    -1,    -1,    -1,    -1,
      -1,   289,    -1,    -1,    -1,    -1,    -1,    -1,   296,    -1,
      -1,    -1,   300,   301,     3,     4,     5,    -1,    -1,     8,
     308,   309,    -1,    12,    13,    14,   314,    16,    -1,    18,
      19,    20,    21,    -1,   322,    24,    -1,    -1,    -1,    -1,
      29,    -1,    -1,    -1,    33,    34,    35,    36,    37,    38,
      39,    -1,    -1,    42,    -1,    -1,    -1,    -1,    47,    -1,
      49,    -1,    51,    -1,   352,    -1,    -1,   355,    57,    -1,
      59,    60,    -1,    62,    -1,    64,    65,    66,    -1,    68,
      69,    70,    71,    -1,    -1,    74,    75,    76,    77,    78,
      79,    80,    -1,    -1,    83,    84,    85,    86,    87,     3,
       4,     5,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    16,    -1,    18,    19,    20,    21,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    29,    -1,    -1,    -1,    33,
      34,    35,    36,    -1,    38,    39,    -1,    -1,    42,    -1,
      -1,    -1,    -1,    -1,    -1,    49,    -1,    51,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    59,    60,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    69,    70,    71,    -1,    -1,
      -1,    75,    -1,    77,    78,    79,    80,    -1,     6,    83,
      84,     9,    86,    87,    -1,    -1,    -1,    15,    -1,    17,
      -1,    -1,    -1,    -1,    22,    23,    -1,    25,    26,    27,
      -1,    -1,    -1,    31,    32,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    40,    -1,    -1,    43,    44,    45,    46,    -1,
       6,    -1,    -1,     9,    -1,    53,    54,    55,    56,    15,
      58,    17,    -1,    61,    -1,    63,    22,    23,    -1,    25,
      26,    27,    -1,    -1,    -1,    31,    32,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    40,    -1,    -1,    43,    44,    45,
      46,    -1,    -1,    -1,    -1,    -1,    -1,    53,    54,    55,
      56,     6,    58,    -1,     9,    61,    11,    63,    -1,    -1,
      15,    -1,    17,    -1,    -1,    -1,    -1,    22,    23,    -1,
      25,    26,    27,    -1,    -1,    -1,    31,    32,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    40,    -1,    -1,    43,    44,
      45,    46,    -1,     6,    -1,    -1,     9,    -1,    -1,    54,
      55,    56,    15,    58,    17,    -1,    61,    -1,    63,    22,
      23,    -1,    25,    26,    27,    -1,    -1,    -1,    31,    32,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    40,    -1,    -1,
      43,    44,    45,    46,    -1,     6,    -1,    -1,     9,    52,
      -1,    54,    55,    56,    15,    58,    17,    -1,    61,    -1,
      63,    22,    23,    -1,    25,    26,    27,    -1,    -1,    -1,
      31,    32,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    40,
      -1,    -1,    43,    44,    45,    46,    -1,     6,    -1,    -1,
       9,    -1,    53,    54,    55,    56,    15,    58,    17,    -1,
      61,    -1,    63,    22,    23,    -1,    25,    26,    27,    -1,
      -1,    -1,    31,    32,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    40,    -1,    -1,    43,    44,    45,    46,    -1,     6,
      -1,    -1,     9,    52,    -1,    54,    55,    56,    15,    58,
      17,    -1,    61,    -1,    63,    22,    23,    -1,    25,    26,
      27,    -1,    -1,    -1,    31,    32,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    40,    -1,    -1,    43,    44,    45,    46,
      -1,     6,    -1,    -1,     9,    52,    11,    54,    55,    56,
      15,    58,    17,    -1,    61,    -1,    63,    22,    23,    -1,
      25,    26,    27,    -1,    -1,    -1,    31,    32,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    40,    -1,    -1,    43,    44,
      45,    46,    -1,     6,    -1,    -1,     9,    -1,    -1,    54,
      55,    56,    15,    58,    17,    -1,    61,    -1,    63,    22,
      23,    -1,    25,    26,    27,    -1,    -1,    -1,    31,    32,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    40,    -1,    -1,
      43,    44,    45,    46,    -1,     6,    -1,    -1,     9,    -1,
      53,    54,    55,    56,    15,    58,    17,    -1,    61,    -1,
      63,    22,    23,    -1,    25,    26,    27,    -1,    -1,    -1,
      31,    32,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    40,
      -1,    -1,    43,    44,    45,    46,    -1,     6,    -1,    -1,
       9,    52,    -1,    54,    55,    56,    15,    58,    17,    -1,
      61,    -1,    63,    22,    23,    -1,    25,    26,    27,    -1,
      -1,    -1,    31,    32,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    40,    -1,    -1,    43,    44,    45,    46,    -1,     6,
      -1,    -1,     9,    52,    11,    54,    55,    56,    15,    58,
      17,    -1,    61,    -1,    63,    22,    23,    -1,    25,    26,
      27,    -1,    -1,    -1,    31,    32,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    40,    -1,    -1,    43,    44,    45,    46,
      -1,     6,    -1,    -1,     9,    -1,    11,    54,    55,    56,
      15,    58,    17,    -1,    61,    -1,    63,    22,    23,    -1,
      25,    26,    27,    -1,    -1,    -1,    31,    32,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    40,    -1,    -1,    43,    44,
      45,    46,    -1,     6,    -1,    -1,     9,    -1,    -1,    54,
      55,    56,    15,    58,    17,    -1,    61,    -1,    63,    22,
      23,    -1,    25,    26,    27,    -1,    -1,    -1,    31,    32,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    40,    -1,    -1,
      43,    44,    45,    46,    -1,     6,    -1,    -1,     9,    52,
      11,    54,    55,    56,    15,    58,    17,    -1,    61,    -1,
      63,    22,    23,    -1,    25,    26,    27,    -1,    -1,    -1,
      31,    32,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    40,
      -1,    -1,    43,    44,    45,    46,    -1,     6,    -1,    -1,
       9,    -1,    11,    54,    55,    56,    15,    58,    17,    -1,
      61,    -1,    63,    22,    23,    -1,    25,    26,    27,    -1,
      -1,    -1,    31,    32,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    40,    -1,    -1,    43,    44,    45,    46,    -1,     6,
      -1,    -1,     9,    -1,    -1,    54,    55,    56,    15,    58,
      17,    -1,    61,    -1,    63,    22,    23,    -1,    25,    26,
      27,    -1,    -1,    -1,    31,    32,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    40,    -1,    -1,    43,    44,    45,    46,
      -1,     6,    -1,    -1,     9,    52,    11,    54,    55,    56,
      15,    58,    17,    -1,    61,    -1,    63,    22,    23,    -1,
      25,    26,    27,    -1,    -1,    -1,    31,    32,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    40,    -1,    -1,    43,    44,
      45,    46,    -1,     6,    -1,    -1,     9,    -1,    11,    54,
      55,    56,    15,    58,    17,    -1,    61,    -1,    63,    22,
      23,    -1,    25,    26,    27,    -1,    -1,    -1,    31,    32,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    40,    -1,    -1,
      43,    44,    45,    46,    -1,     6,    -1,    -1,     9,    -1,
      -1,    54,    55,    56,    15,    58,    17,    -1,    61,    -1,
      63,    22,    23,    -1,    25,    26,    27,    -1,    -1,    -1,
      31,    32,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    40,
      -1,    -1,    43,    44,    45,    46,    -1,     6,    -1,    -1,
       9,    -1,    53,    54,    55,    56,    15,    58,    17,    -1,
      61,    -1,    63,    22,    23,    -1,    25,    26,    27,    -1,
      -1,    -1,    31,    32,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    40,    -1,    -1,    43,    44,    45,    46,    -1,     6,
      -1,    -1,     9,    -1,    53,    54,    55,    56,    15,    58,
      17,    -1,    61,    -1,    63,    22,    23,    -1,    25,    26,
      27,    -1,    -1,    -1,    31,    32,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    40,    -1,    -1,    43,    44,    45,    46,
      -1,     6,    -1,    -1,     9,    52,    11,    54,    55,    56,
      15,    58,    17,    -1,    61,    -1,    63,    22,    23,    -1,
      25,    26,    27,    -1,    -1,    -1,    31,    32,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    40,    -1,    -1,    43,    44,
      45,    46,    -1,     6,    -1,    -1,     9,    -1,    -1,    54,
      55,    56,    15,    58,    17,    -1,    61,    -1,    63,    22,
      23,    -1,    25,    26,    27,    -1,    -1,    -1,    31,    32,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    40,    -1,    -1,
      43,    44,    45,    46,    -1,     6,    -1,    -1,     9,    -1,
      53,    54,    55,    56,    15,    58,    17,    -1,    61,    -1,
      63,    22,    23,    -1,    25,    26,    27,    -1,    -1,    -1,
      31,    32,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    40,
      -1,    -1,    43,    44,    45,    46,    -1,     6,    -1,    -1,
       9,    -1,    53,    54,    55,    56,    15,    58,    17,    -1,
      61,    -1,    63,    22,    23,    -1,    25,    26,    27,    -1,
      -1,    -1,    31,    32,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    40,    -1,    -1,    43,    44,    45,    46,    -1,     6,
      -1,    -1,     9,    -1,    53,    54,    55,    56,    15,    58,
      17,    -1,    61,    -1,    63,    22,    23,    -1,    25,    26,
      27,    -1,    -1,    -1,    31,    32,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    40,    -1,    -1,    43,    44,    45,    46,
      -1,     6,    -1,    -1,     9,    -1,    53,    54,    55,    56,
      15,    58,    17,    -1,    61,    -1,    63,    22,    23,    -1,
      25,    26,    27,    -1,    -1,    -1,    31,    32,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    40,    -1,    -1,    43,    44,
      45,    46,    -1,     6,    -1,    -1,     9,    -1,    53,    54,
      55,    56,    15,    58,    17,    -1,    61,    -1,    63,    22,
      23,    -1,    25,    26,    27,    -1,    -1,    -1,    31,    32,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    40,    -1,    -1,
      43,    44,    45,    46,    -1,     6,    -1,    -1,     9,    -1,
      53,    54,    55,    56,    15,    58,    17,    -1,    61,    -1,
      63,    22,    23,    -1,    25,    26,    27,    -1,    -1,    -1,
      31,    32,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    40,
      -1,    -1,    43,    44,    45,    46,    -1,     6,    -1,    -1,
       9,    -1,    -1,    54,    55,    56,    15,    58,    17,    -1,
      61,    -1,    63,    22,    23,    -1,    25,    -1,    27,    -1,
      -1,    -1,    31,    32,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    40,    -1,    -1,    43,    44,    45,    46,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    54,    55,    56,    15,    58,
      17,    -1,    61,    -1,    63,    22,    23,    -1,    -1,    26,
      27,    -1,    -1,    -1,    31,    32,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    40,    -1,    -1,    43,    44,    -1,    46,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    54,    55,    56,
      15,    58,    17,    -1,    61,    -1,    63,    22,    23,    -1,
      -1,    26,    27,    -1,    -1,    -1,    31,    32,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    40,    -1,    -1,    43,    44,
      -1,    46,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    54,
      55,    56,    -1,    58,    -1,    -1,    61,    -1,    63
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    72,    73,    81,    82,    91,    92,    53,    53,    53,
      53,     0,     3,     4,     5,     8,    12,    13,    14,    16,
      18,    19,    20,    21,    24,    29,    33,    34,    35,    36,
      37,    38,    39,    42,    47,    49,    51,    57,    59,    60,
      62,    64,    65,    66,    68,    69,    70,    71,    74,    75,
      76,    77,    78,    79,    80,    83,    84,    85,    86,    87,
      93,    94,    95,   105,   105,   107,    30,   104,   104,    59,
      77,    95,   105,    36,    36,    36,   107,   104,   104,   105,
     108,   109,    28,    95,   100,   101,    98,    99,   105,    98,
      98,    95,   105,    95,    95,    95,   105,    36,    36,    88,
     118,    36,   104,   105,   110,   111,   107,   104,   104,    88,
      36,    36,    36,   105,    36,    95,    36,   105,    36,    36,
      36,    36,    36,   105,    93,     6,     9,    15,    17,    22,
      23,    25,    26,    27,    31,    32,    40,    43,    44,    45,
      46,    53,    54,    55,    56,    58,    61,    63,    36,    67,
     112,    10,   105,   106,   107,   107,    53,    17,   105,   105,
      95,    10,   107,   107,     7,    53,    11,    11,    11,    48,
      26,    11,    17,    26,    26,    52,    17,    36,    17,    36,
      95,    95,   105,    95,    95,   107,    10,    53,    11,    53,
     107,   107,    53,    88,    88,    95,    53,    95,    53,    95,
      17,    95,    95,    95,    95,    95,    53,    95,    95,    86,
      95,    95,    95,    95,    95,    95,    95,    95,    36,    40,
      86,    96,   105,    95,    95,    95,    96,    95,    96,    96,
      95,    95,    95,   102,   103,    95,    11,    95,    11,    50,
      10,    10,    96,    52,    52,    52,    95,    10,    10,    36,
     108,   101,   101,    95,    98,    95,    95,    95,    95,    52,
      62,    64,    65,    66,    97,   105,    95,    52,    97,    52,
      11,    52,    53,    52,    53,    86,   105,   110,    53,    53,
      11,   113,    11,    52,    11,    11,   105,   114,    52,    11,
      11,    52,    11,    96,    96,    40,    41,    46,    55,    56,
      41,    11,    52,   107,   106,    95,    95,    53,    17,    17,
      95,    95,   105,    53,    17,   105,   105,   105,   105,    52,
     112,    53,    17,    52,   104,   105,    52,    95,   106,   105,
      11,    36,    53,    95,   105,   105,    52,    96,    95,    96,
      96,    96,    95,   103,    95,    95,    11,    95,    11,    11,
     112,   112,    17,    11,    95,    17,    52,     7,    52,    52,
      52,   114,    52,   105,   115,    11,    11,    10,    53,    53,
     105,    53,    97,    97,    11,    11,    95,    97,    53,    95,
     105,    11,    10,    52,   105,   105,   105,   116,    52,    97,
      97,    53,    53,   113,   114,   105,    11,    11,    11,    36,
      52,    11,   114,   105,   105,    52,   116,   117,   115,    52,
      52,    11,    52,   117
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    90,    91,    92,    92,    92,    92,    92,    93,    93,
      94,    94,    94,    94,    94,    94,    94,    94,    94,    94,
      94,    94,    94,    94,    94,    94,    94,    94,    94,    94,
      94,    94,    94,    95,    95,    95,    95,    95,    95,    95,
      95,    95,    95,    95,    95,    95,    95,    95,    95,    95,
      95,    95,    95,    95,    95,    95,    95,    95,    95,    95,
      95,    95,    95,    95,    95,    95,    95,    95,    95,    95,
      95,    95,    95,    95,    95,    95,    95,    95,    95,    95,
      95,    95,    95,    95,    95,    95,    95,    95,    95,    95,
      95,    95,    95,    96,    96,    96,    96,    96,    96,    96,
      96,    97,    97,    97,    97,    97,    97,    97,    97,    97,
      97,    98,    98,    99,   100,   100,   101,   101,   101,   101,
     102,   102,   103,   103,   104,   104,   105,   106,   106,   107,
     107,   108,   108,   109,   110,   110,   111,   111,   111,   112,
     112,   113,   113,   114,   114,   114,   114,   114,   114,   115,
     115,   116,   116,   116,   117,   117,   118,   118
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     2,     2,     2,     2,     2,     0,     2,     1,
       3,     3,     3,     7,     7,     5,     3,     4,     4,     4,
       8,     5,     7,     8,     5,     7,     2,     4,     3,     3,
       3,     3,     5,     1,     3,     3,     3,     3,     2,     2,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       2,     4,     1,     3,     2,     4,     5,     5,     4,     5,
       5,     4,     4,     4,     4,     1,     1,     6,     4,     5,
       5,     3,     3,     3,     3,     1,     1,     3,     3,     3,
       3,     5,     6,     4,     6,    10,    10,     4,     6,     4,
       8,     4,     4,     3,     3,     3,     3,     2,     1,     1,
       3,     4,     5,     5,     4,     4,     2,     3,     3,     2,
       2,     3,     1,     3,     1,     0,     3,     3,     1,     1,
       1,     0,     3,     1,     3,     0,     1,     3,     1,     4,
       2,     3,     1,     7,     3,     1,     3,     3,     1,     2,
       0,     5,     0,     6,     5,     3,     4,     3,     1,     5,
       3,     4,     3,     1,     3,     1,     1,     0
};


#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)
#define YYEMPTY         (-2)
#define YYEOF           0

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                  \
do                                                              \
  if (yychar == YYEMPTY)                                        \
    {                                                           \
      yychar = (Token);                                         \
      yylval = (Value);                                         \
      YYPOPSTACK (yylen);                                       \
      yystate = *yyssp;                                         \
      goto yybackup;                                            \
    }                                                           \
  else                                                          \
    {                                                           \
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;                                                  \
    }                                                           \
while (0)

/* Error token number */
#define YYTERROR        1
#define YYERRCODE       256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)                                \
    do                                                                  \
      if (N)                                                            \
        {                                                               \
          (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;        \
          (Current).first_column = YYRHSLOC (Rhs, 1).first_column;      \
          (Current).last_line    = YYRHSLOC (Rhs, N).last_line;         \
          (Current).last_column  = YYRHSLOC (Rhs, N).last_column;       \
        }                                                               \
      else                                                              \
        {                                                               \
          (Current).first_line   = (Current).last_line   =              \
            YYRHSLOC (Rhs, 0).last_line;                                \
          (Current).first_column = (Current).last_column =              \
            YYRHSLOC (Rhs, 0).last_column;                              \
        }                                                               \
    while (0)
#endif

#define YYRHSLOC(Rhs, K) ((Rhs)[K])


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL

/* Print *YYLOCP on YYO.  Private, do not rely on its existence. */

YY_ATTRIBUTE_UNUSED
static unsigned
yy_location_print_ (FILE *yyo, YYLTYPE const * const yylocp)
{
  unsigned res = 0;
  int end_col = 0 != yylocp->last_column ? yylocp->last_column - 1 : 0;
  if (0 <= yylocp->first_line)
    {
      res += YYFPRINTF (yyo, "%d", yylocp->first_line);
      if (0 <= yylocp->first_column)
        res += YYFPRINTF (yyo, ".%d", yylocp->first_column);
    }
  if (0 <= yylocp->last_line)
    {
      if (yylocp->first_line < yylocp->last_line)
        {
          res += YYFPRINTF (yyo, "-%d", yylocp->last_line);
          if (0 <= end_col)
            res += YYFPRINTF (yyo, ".%d", end_col);
        }
      else if (0 <= end_col && yylocp->first_column < end_col)
        res += YYFPRINTF (yyo, "-%d", end_col);
    }
  return res;
 }

#  define YY_LOCATION_PRINT(File, Loc)          \
  yy_location_print_ (File, &(Loc))

# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


# define YY_SYMBOL_PRINT(Title, Type, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Type, Value, Location); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*----------------------------------------.
| Print this symbol's value on YYOUTPUT.  |
`----------------------------------------*/

static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
{
  FILE *yyo = yyoutput;
  YYUSE (yyo);
  YYUSE (yylocationp);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# endif
  YYUSE (yytype);
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
{
  YYFPRINTF (yyoutput, "%s %s (",
             yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  YY_LOCATION_PRINT (yyoutput, *yylocationp);
  YYFPRINTF (yyoutput, ": ");
  yy_symbol_value_print (yyoutput, yytype, yyvaluep, yylocationp);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yytype_int16 *yyssp, YYSTYPE *yyvsp, YYLTYPE *yylsp, int yyrule)
{
  unsigned long int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       yystos[yyssp[yyi + 1 - yynrhs]],
                       &(yyvsp[(yyi + 1) - (yynrhs)])
                       , &(yylsp[(yyi + 1) - (yynrhs)])                       );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, yylsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
static YYSIZE_T
yystrlen (const char *yystr)
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy (char *yydest, const char *yysrc)
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
        switch (*++yyp)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++yyp != '\\')
              goto do_not_strip_quotes;
            /* Fall through.  */
          default:
            if (yyres)
              yyres[yyn] = *yyp;
            yyn++;
            break;

          case '"':
            if (yyres)
              yyres[yyn] = '\0';
            return yyn;
          }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYSIZE_T *yymsg_alloc, char **yymsg,
                yytype_int16 *yyssp, int yytoken)
{
  YYSIZE_T yysize0 = yytnamerr (YY_NULLPTR, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[*yyssp];
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;

          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                {
                  YYSIZE_T yysize1 = yysize + yytnamerr (YY_NULLPTR, yytname[yyx]);
                  if (! (yysize <= yysize1
                         && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                    return 2;
                  yysize = yysize1;
                }
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
    default: /* Avoid compiler warnings. */
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  {
    YYSIZE_T yysize1 = yysize + yystrlen (yyformat);
    if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
      return 2;
    yysize = yysize1;
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          yyp++;
          yyformat++;
        }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep, YYLTYPE *yylocationp)
{
  YYUSE (yyvaluep);
  YYUSE (yylocationp);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yytype);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}




/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Location data for the lookahead symbol.  */
YYLTYPE yylloc
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
  = { 1, 1, 1, 1 }
# endif
;
/* Number of syntax errors so far.  */
int yynerrs;


/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       'yyss': related to states.
       'yyvs': related to semantic values.
       'yyls': related to locations.

       Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    /* The location stack.  */
    YYLTYPE yylsa[YYINITDEPTH];
    YYLTYPE *yyls;
    YYLTYPE *yylsp;

    /* The locations where the error started and ended.  */
    YYLTYPE yyerror_range[3];

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;
  YYLTYPE yyloc;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N), yylsp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yylsp = yyls = yylsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */
  yylsp[0] = yylloc;
  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        YYSTYPE *yyvs1 = yyvs;
        yytype_int16 *yyss1 = yyss;
        YYLTYPE *yyls1 = yyls;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * sizeof (*yyssp),
                    &yyvs1, yysize * sizeof (*yyvsp),
                    &yyls1, yysize * sizeof (*yylsp),
                    &yystacksize);

        yyls = yyls1;
        yyss = yyss1;
        yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yytype_int16 *yyss1 = yyss;
        union yyalloc *yyptr =
          (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
        YYSTACK_RELOCATE (yyls_alloc, yyls);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;
      yylsp = yyls + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
                  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END
  *++yylsp = yylloc;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];

  /* Default location. */
  YYLLOC_DEFAULT (yyloc, (yylsp - yylen), yylen);
  yyerror_range[1] = yyloc;
  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:
#line 148 "parser.ypp" /* yacc.c:1648  */
    {untypedAST = new MonaUntypedAST((yyvsp[0].declarationList));}
#line 2169 "parser.cpp" /* yacc.c:1648  */
    break;

  case 3:
#line 152 "parser.ypp" /* yacc.c:1648  */
    {options.m2l = false; options.mode = LINEAR;}
#line 2175 "parser.cpp" /* yacc.c:1648  */
    break;

  case 4:
#line 154 "parser.ypp" /* yacc.c:1648  */
    {options.m2l = false; options.mode = TREE;}
#line 2181 "parser.cpp" /* yacc.c:1648  */
    break;

  case 5:
#line 156 "parser.ypp" /* yacc.c:1648  */
    {options.m2l = true; options.mode = LINEAR;}
#line 2187 "parser.cpp" /* yacc.c:1648  */
    break;

  case 6:
#line 158 "parser.ypp" /* yacc.c:1648  */
    {options.m2l = true; options.mode = TREE;}
#line 2193 "parser.cpp" /* yacc.c:1648  */
    break;

  case 7:
#line 160 "parser.ypp" /* yacc.c:1648  */
    {options.m2l = false; options.mode = LINEAR;}
#line 2199 "parser.cpp" /* yacc.c:1648  */
    break;

  case 8:
#line 164 "parser.ypp" /* yacc.c:1648  */
    {if ((yyvsp[-1].declaration)) (yyvsp[0].declarationList)->push_front((yyvsp[-1].declaration)); (yyval.declarationList) = (yyvsp[0].declarationList);}
#line 2205 "parser.cpp" /* yacc.c:1648  */
    break;

  case 9:
#line 166 "parser.ypp" /* yacc.c:1648  */
    {(yyval.declarationList) = new DeclarationList(); 
	         if ((yyvsp[0].declaration)) (yyval.declarationList)->push_front((yyvsp[0].declaration));}
#line 2212 "parser.cpp" /* yacc.c:1648  */
    break;

  case 10:
#line 171 "parser.ypp" /* yacc.c:1648  */
    {(yyval.declaration) = new Assertion_Declaration((yyvsp[-1].untypedExp), POS((yylsp[-2])));}
#line 2218 "parser.cpp" /* yacc.c:1648  */
    break;

  case 11:
#line 173 "parser.ypp" /* yacc.c:1648  */
    {(yyval.declaration) = new Guide_Declaration((yyvsp[-1].guideFuncList), POS((yylsp[-2])));}
#line 2224 "parser.cpp" /* yacc.c:1648  */
    break;

  case 12:
#line 175 "parser.ypp" /* yacc.c:1648  */
    {(yyval.declaration) = new Universe_Declaration((yyvsp[-1].univList), POS((yylsp[-2])));
		 anyUniverses = true;}
#line 2231 "parser.cpp" /* yacc.c:1648  */
    break;

  case 13:
#line 178 "parser.ypp" /* yacc.c:1648  */
    {(yyval.declaration) = new Default_Declaration(vVar1, (yyvsp[-4].name), (yyvsp[-1].untypedExp), POS((yylsp[-6])));}
#line 2237 "parser.cpp" /* yacc.c:1648  */
    break;

  case 14:
#line 180 "parser.ypp" /* yacc.c:1648  */
    {(yyval.declaration) = new Default_Declaration(vVar2, (yyvsp[-4].name), (yyvsp[-1].untypedExp), POS((yylsp[-6])));}
#line 2243 "parser.cpp" /* yacc.c:1648  */
    break;

  case 15:
#line 182 "parser.ypp" /* yacc.c:1648  */
    {(yyval.declaration) = new Constant_Declaration((yyvsp[-3].name), (yyvsp[-1].arithExp), POS((yylsp[-4])));}
#line 2249 "parser.cpp" /* yacc.c:1648  */
    break;

  case 16:
#line 184 "parser.ypp" /* yacc.c:1648  */
    {(yyval.declaration) = new Variable_Declaration(vVar0, NULL, (yyvsp[-1].varDeclList), POS((yylsp[-2])));}
#line 2255 "parser.cpp" /* yacc.c:1648  */
    break;

  case 17:
#line 186 "parser.ypp" /* yacc.c:1648  */
    {(yyval.declaration) = new Variable_Declaration(vVar1, (yyvsp[-2].nameList), (yyvsp[-1].varDeclList), POS((yylsp[-3])));}
#line 2261 "parser.cpp" /* yacc.c:1648  */
    break;

  case 18:
#line 188 "parser.ypp" /* yacc.c:1648  */
    {(yyval.declaration) = new Variable_Declaration(vVar2, (yyvsp[-2].nameList), (yyvsp[-1].varDeclList), POS((yylsp[-3])));}
#line 2267 "parser.cpp" /* yacc.c:1648  */
    break;

  case 19:
#line 190 "parser.ypp" /* yacc.c:1648  */
    {(yyval.declaration) = new Variable_Declaration(vTree, (yyvsp[-2].nameList), (yyvsp[-1].varDeclList), POS((yylsp[-3])));}
#line 2273 "parser.cpp" /* yacc.c:1648  */
    break;

  case 20:
#line 192 "parser.ypp" /* yacc.c:1648  */
    {(yyval.declaration) = new Predicate_Declaration((yyvsp[-6].name), (yyvsp[-4].parDeclList), (yyvsp[-1].untypedExp), POS((yylsp[-7])));}
#line 2279 "parser.cpp" /* yacc.c:1648  */
    break;

  case 21:
#line 194 "parser.ypp" /* yacc.c:1648  */
    {(yyval.declaration) = new Predicate_Declaration((yyvsp[-3].name), new ParDeclList(), 
						(yyvsp[-1].untypedExp), POS((yylsp[-4])));}
#line 2286 "parser.cpp" /* yacc.c:1648  */
    break;

  case 22:
#line 197 "parser.ypp" /* yacc.c:1648  */
    {(yyval.declaration) = new Predicate_Declaration((yyvsp[-5].name), new ParDeclList(), 
						(yyvsp[-1].untypedExp), POS((yylsp[-6])));}
#line 2293 "parser.cpp" /* yacc.c:1648  */
    break;

  case 23:
#line 200 "parser.ypp" /* yacc.c:1648  */
    {(yyval.declaration) = new Macro_Declaration((yyvsp[-6].name), (yyvsp[-4].parDeclList), (yyvsp[-1].untypedExp), POS((yylsp[-7])));}
#line 2299 "parser.cpp" /* yacc.c:1648  */
    break;

  case 24:
#line 202 "parser.ypp" /* yacc.c:1648  */
    {(yyval.declaration) = new Macro_Declaration((yyvsp[-3].name), new ParDeclList(), 
					    (yyvsp[-1].untypedExp), POS((yylsp[-4])));}
#line 2306 "parser.cpp" /* yacc.c:1648  */
    break;

  case 25:
#line 205 "parser.ypp" /* yacc.c:1648  */
    {(yyval.declaration) = new Macro_Declaration((yyvsp[-5].name), new ParDeclList(), 
					    (yyvsp[-1].untypedExp), POS((yylsp[-6])));}
#line 2313 "parser.cpp" /* yacc.c:1648  */
    break;

  case 26:
#line 208 "parser.ypp" /* yacc.c:1648  */
    {(yyval.declaration) = new Expression_Declaration((yyvsp[-1].untypedExp), POS((yylsp[-1])));}
#line 2319 "parser.cpp" /* yacc.c:1648  */
    break;

  case 27:
#line 210 "parser.ypp" /* yacc.c:1648  */
    {(yyval.declaration) = new Verify_Declaration((yyvsp[-2].string), (yyvsp[-1].untypedExp), POS((yylsp[-3])));}
#line 2325 "parser.cpp" /* yacc.c:1648  */
    break;

  case 28:
#line 212 "parser.ypp" /* yacc.c:1648  */
    {(yyval.declaration) = new Execute_Declaration((yyvsp[-1].untypedExp), POS((yylsp[-2])));}
#line 2331 "parser.cpp" /* yacc.c:1648  */
    break;

  case 29:
#line 214 "parser.ypp" /* yacc.c:1648  */
    {loadFile((yyvsp[-1].string)+1); (yyval.declaration) = 0;}
#line 2337 "parser.cpp" /* yacc.c:1648  */
    break;

  case 30:
#line 216 "parser.ypp" /* yacc.c:1648  */
    {(yyval.declaration) = new LastPos_Declaration((yyvsp[-1].name), POS((yylsp[-2])));}
#line 2343 "parser.cpp" /* yacc.c:1648  */
    break;

  case 31:
#line 218 "parser.ypp" /* yacc.c:1648  */
    {(yyval.declaration) = new AllPos_Declaration((yyvsp[-1].name), POS((yylsp[-2])));}
#line 2349 "parser.cpp" /* yacc.c:1648  */
    break;

  case 32:
#line 220 "parser.ypp" /* yacc.c:1648  */
    {(yyval.declaration) = new Type_Declaration((yyvsp[-3].name), (yyvsp[-1].variantList), POS((yylsp[-4])));}
#line 2355 "parser.cpp" /* yacc.c:1648  */
    break;

  case 33:
#line 224 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Name((yyvsp[0].name), POS((yylsp[0])));}
#line 2361 "parser.cpp" /* yacc.c:1648  */
    break;

  case 34:
#line 226 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = (yyvsp[-1].untypedExp);}
#line 2367 "parser.cpp" /* yacc.c:1648  */
    break;

  case 35:
#line 228 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Sub((yyvsp[-2].untypedExp), (yyvsp[0].untypedExp), POS((yylsp[-1])));}
#line 2373 "parser.cpp" /* yacc.c:1648  */
    break;

  case 36:
#line 230 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_In((yyvsp[-2].untypedExp), (yyvsp[0].untypedExp), POS((yylsp[-1])));}
#line 2379 "parser.cpp" /* yacc.c:1648  */
    break;

  case 37:
#line 232 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_NotIn((yyvsp[-2].untypedExp), (yyvsp[0].untypedExp), POS((yylsp[-1])));}
#line 2385 "parser.cpp" /* yacc.c:1648  */
    break;

  case 38:
#line 234 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Min((yyvsp[0].untypedExp), POS((yylsp[-1])));}
#line 2391 "parser.cpp" /* yacc.c:1648  */
    break;

  case 39:
#line 236 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Max((yyvsp[0].untypedExp), POS((yylsp[-1])));}
#line 2397 "parser.cpp" /* yacc.c:1648  */
    break;

  case 40:
#line 238 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Less((yyvsp[-2].untypedExp), (yyvsp[0].untypedExp), POS((yylsp[-1])));}
#line 2403 "parser.cpp" /* yacc.c:1648  */
    break;

  case 41:
#line 240 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_LessEq((yyvsp[-2].untypedExp), (yyvsp[0].untypedExp), POS((yylsp[-1])));}
#line 2409 "parser.cpp" /* yacc.c:1648  */
    break;

  case 42:
#line 242 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_GreaterEq((yyvsp[-2].untypedExp), (yyvsp[0].untypedExp), POS((yylsp[-1])));}
#line 2415 "parser.cpp" /* yacc.c:1648  */
    break;

  case 43:
#line 244 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Greater((yyvsp[-2].untypedExp), (yyvsp[0].untypedExp), POS((yylsp[-1])));}
#line 2421 "parser.cpp" /* yacc.c:1648  */
    break;

  case 44:
#line 246 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Equal((yyvsp[-2].untypedExp), (yyvsp[0].untypedExp), POS((yylsp[-1])));}
#line 2427 "parser.cpp" /* yacc.c:1648  */
    break;

  case 45:
#line 248 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_NotEqual((yyvsp[-2].untypedExp), (yyvsp[0].untypedExp), POS((yylsp[-1])));}
#line 2433 "parser.cpp" /* yacc.c:1648  */
    break;

  case 46:
#line 250 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Impl((yyvsp[-2].untypedExp), (yyvsp[0].untypedExp), POS((yylsp[-1])));}
#line 2439 "parser.cpp" /* yacc.c:1648  */
    break;

  case 47:
#line 252 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Biimpl((yyvsp[-2].untypedExp), (yyvsp[0].untypedExp), POS((yylsp[-1])));}
#line 2445 "parser.cpp" /* yacc.c:1648  */
    break;

  case 48:
#line 254 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_And((yyvsp[-2].untypedExp), (yyvsp[0].untypedExp), POS((yylsp[-1])));}
#line 2451 "parser.cpp" /* yacc.c:1648  */
    break;

  case 49:
#line 256 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Or((yyvsp[-2].untypedExp), (yyvsp[0].untypedExp), POS((yylsp[-1])));}
#line 2457 "parser.cpp" /* yacc.c:1648  */
    break;

  case 50:
#line 258 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Not((yyvsp[0].untypedExp), POS((yylsp[-1])));}
#line 2463 "parser.cpp" /* yacc.c:1648  */
    break;

  case 51:
#line 260 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Root((yyvsp[-1].name), POS((yylsp[-3])));}
#line 2469 "parser.cpp" /* yacc.c:1648  */
    break;

  case 52:
#line 262 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Root(NULL, POS((yylsp[0])));}
#line 2475 "parser.cpp" /* yacc.c:1648  */
    break;

  case 53:
#line 264 "parser.ypp" /* yacc.c:1648  */
    {check_bits((yyvsp[0].string));
		 (yyval.untypedExp) = new UntypedExp_Dot((yyvsp[-2].untypedExp), (yyvsp[0].string), POS((yylsp[-1])));}
#line 2482 "parser.cpp" /* yacc.c:1648  */
    break;

  case 54:
#line 267 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Up((yyvsp[-1].untypedExp), POS((yylsp[0])));}
#line 2488 "parser.cpp" /* yacc.c:1648  */
    break;

  case 55:
#line 269 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Ex0((yyvsp[-2].varDeclList), (yyvsp[0].untypedExp), POS((yylsp[-3])));}
#line 2494 "parser.cpp" /* yacc.c:1648  */
    break;

  case 56:
#line 271 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Ex1((yyvsp[-3].nameList), (yyvsp[-2].varDeclList), (yyvsp[0].untypedExp), POS((yylsp[-4])));}
#line 2500 "parser.cpp" /* yacc.c:1648  */
    break;

  case 57:
#line 273 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Ex2((yyvsp[-3].nameList), (yyvsp[-2].varDeclList), (yyvsp[0].untypedExp), POS((yylsp[-4])));}
#line 2506 "parser.cpp" /* yacc.c:1648  */
    break;

  case 58:
#line 275 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_All0((yyvsp[-2].varDeclList), (yyvsp[0].untypedExp), POS((yylsp[-3])));}
#line 2512 "parser.cpp" /* yacc.c:1648  */
    break;

  case 59:
#line 277 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_All1((yyvsp[-3].nameList), (yyvsp[-2].varDeclList), (yyvsp[0].untypedExp), POS((yylsp[-4])));}
#line 2518 "parser.cpp" /* yacc.c:1648  */
    break;

  case 60:
#line 279 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_All2((yyvsp[-3].nameList), (yyvsp[-2].varDeclList), (yyvsp[0].untypedExp), POS((yylsp[-4])));}
#line 2524 "parser.cpp" /* yacc.c:1648  */
    break;

  case 61:
#line 281 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Let0((yyvsp[-2].bindExpList), (yyvsp[0].untypedExp), POS((yylsp[-3])));}
#line 2530 "parser.cpp" /* yacc.c:1648  */
    break;

  case 62:
#line 283 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Let1((yyvsp[-2].bindExpList), (yyvsp[0].untypedExp), POS((yylsp[-3])));}
#line 2536 "parser.cpp" /* yacc.c:1648  */
    break;

  case 63:
#line 285 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Let2((yyvsp[-2].bindExpList), (yyvsp[0].untypedExp), POS((yylsp[-3])));}
#line 2542 "parser.cpp" /* yacc.c:1648  */
    break;

  case 64:
#line 287 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Call((yyvsp[-3].name), (yyvsp[-1].untypedExpList), POS((yylsp[-3])));}
#line 2548 "parser.cpp" /* yacc.c:1648  */
    break;

  case 65:
#line 289 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_True(POS((yylsp[0])));}
#line 2554 "parser.cpp" /* yacc.c:1648  */
    break;

  case 66:
#line 291 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_False(POS((yylsp[0])));}
#line 2560 "parser.cpp" /* yacc.c:1648  */
    break;

  case 67:
#line 293 "parser.ypp" /* yacc.c:1648  */
    {if (!(yyvsp[-1].nameList)) yyerror("parse error");
		 (yyval.untypedExp) = new UntypedExp_RootPred((yyvsp[-3].untypedExp), (yyvsp[-1].nameList), POS((yylsp[-5])));}
#line 2567 "parser.cpp" /* yacc.c:1648  */
    break;

  case 68:
#line 296 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_EmptyPred((yyvsp[-1].untypedExp), POS((yylsp[-3])));}
#line 2573 "parser.cpp" /* yacc.c:1648  */
    break;

  case 69:
#line 298 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_PlusModulo((yyvsp[-4].untypedExp), (yyvsp[-2].arithExp), (yyvsp[0].untypedExp), POS((yylsp[-1])));}
#line 2579 "parser.cpp" /* yacc.c:1648  */
    break;

  case 70:
#line 300 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_MinusModulo((yyvsp[-4].untypedExp), (yyvsp[-2].arithExp), (yyvsp[0].untypedExp), POS((yylsp[-1])));}
#line 2585 "parser.cpp" /* yacc.c:1648  */
    break;

  case 71:
#line 302 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Plus((yyvsp[-2].untypedExp), (yyvsp[0].arithExp), POS((yylsp[-1])));}
#line 2591 "parser.cpp" /* yacc.c:1648  */
    break;

  case 72:
#line 304 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Minus((yyvsp[-2].untypedExp), (yyvsp[0].arithExp), POS((yylsp[-1])));}
#line 2597 "parser.cpp" /* yacc.c:1648  */
    break;

  case 73:
#line 306 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Mult((yyvsp[-2].untypedExp), (yyvsp[0].arithExp), POS((yylsp[-1])));}
#line 2603 "parser.cpp" /* yacc.c:1648  */
    break;

  case 74:
#line 308 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Div((yyvsp[-2].untypedExp), (yyvsp[0].arithExp), POS((yylsp[-1])));}
#line 2609 "parser.cpp" /* yacc.c:1648  */
    break;

  case 75:
#line 310 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Int(atoi((yyvsp[0].string)), POS((yylsp[0])));}
#line 2615 "parser.cpp" /* yacc.c:1648  */
    break;

  case 76:
#line 312 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Empty(POS((yylsp[0])));}
#line 2621 "parser.cpp" /* yacc.c:1648  */
    break;

  case 77:
#line 314 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Set((yyvsp[-1].untypedExpList), POS((yylsp[-2])));}
#line 2627 "parser.cpp" /* yacc.c:1648  */
    break;

  case 78:
#line 316 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Union((yyvsp[-2].untypedExp), (yyvsp[0].untypedExp), POS((yylsp[-1])));}
#line 2633 "parser.cpp" /* yacc.c:1648  */
    break;

  case 79:
#line 318 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Inter((yyvsp[-2].untypedExp), (yyvsp[0].untypedExp), POS((yylsp[-1])));}
#line 2639 "parser.cpp" /* yacc.c:1648  */
    break;

  case 80:
#line 320 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Setminus((yyvsp[-2].untypedExp), (yyvsp[0].untypedExp), POS((yylsp[-1])));}
#line 2645 "parser.cpp" /* yacc.c:1648  */
    break;

  case 81:
#line 322 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Import((yyvsp[-2].string)+1, (yyvsp[-1].mapList), POS((yylsp[-4])));}
#line 2651 "parser.cpp" /* yacc.c:1648  */
    break;

  case 82:
#line 324 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Export((yyvsp[-3].string)+1, (yyvsp[-1].untypedExp), POS((yylsp[-5])));}
#line 2657 "parser.cpp" /* yacc.c:1648  */
    break;

  case 83:
#line 326 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Prefix((yyvsp[-1].untypedExp), POS((yylsp[-3])));}
#line 2663 "parser.cpp" /* yacc.c:1648  */
    break;

  case 84:
#line 328 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_InStateSpace((yyvsp[-3].untypedExp), (yyvsp[-1].nameList), POS((yylsp[-5])));}
#line 2669 "parser.cpp" /* yacc.c:1648  */
    break;

  case 85:
#line 331 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Variant((yyvsp[-7].untypedExp), (yyvsp[-5].untypedExp), (yyvsp[-3].name), (yyvsp[-1].name), POS((yylsp[-9])));}
#line 2675 "parser.cpp" /* yacc.c:1648  */
    break;

  case 86:
#line 334 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Succ((yyvsp[-7].untypedExp), (yyvsp[-5].name), (yyvsp[-3].name), (yyvsp[-1].name), POS((yylsp[-9])));}
#line 2681 "parser.cpp" /* yacc.c:1648  */
    break;

  case 87:
#line 336 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_WellFormedTree((yyvsp[-1].untypedExp), POS((yylsp[-3])));}
#line 2687 "parser.cpp" /* yacc.c:1648  */
    break;

  case 88:
#line 338 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Type((yyvsp[-3].untypedExp), (yyvsp[-1].name), POS((yylsp[-5])));}
#line 2693 "parser.cpp" /* yacc.c:1648  */
    break;

  case 89:
#line 340 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_SomeType((yyvsp[-1].untypedExp), POS((yylsp[-3])));}
#line 2699 "parser.cpp" /* yacc.c:1648  */
    break;

  case 90:
#line 343 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_ConstTree((yyvsp[-5].untypedExp), (yyvsp[-3].name), (yyvsp[-1].constNode), POS((yylsp[-7])));}
#line 2705 "parser.cpp" /* yacc.c:1648  */
    break;

  case 91:
#line 345 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_TreeRoot((yyvsp[-1].untypedExp), POS((yylsp[-3])));}
#line 2711 "parser.cpp" /* yacc.c:1648  */
    break;

  case 92:
#line 347 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = new UntypedExp_Restrict((yyvsp[-1].untypedExp), POS((yylsp[-3])));}
#line 2717 "parser.cpp" /* yacc.c:1648  */
    break;

  case 93:
#line 351 "parser.ypp" /* yacc.c:1648  */
    {(yyval.arithExp) = new ArithExp_Add((yyvsp[-2].arithExp), (yyvsp[0].arithExp), POS((yylsp[-1])));}
#line 2723 "parser.cpp" /* yacc.c:1648  */
    break;

  case 94:
#line 353 "parser.ypp" /* yacc.c:1648  */
    {(yyval.arithExp) = new ArithExp_Subtr((yyvsp[-2].arithExp), (yyvsp[0].arithExp), POS((yylsp[-1])));}
#line 2729 "parser.cpp" /* yacc.c:1648  */
    break;

  case 95:
#line 355 "parser.ypp" /* yacc.c:1648  */
    {(yyval.arithExp) = new ArithExp_Mult((yyvsp[-2].arithExp), (yyvsp[0].arithExp), POS((yylsp[-1])));}
#line 2735 "parser.cpp" /* yacc.c:1648  */
    break;

  case 96:
#line 357 "parser.ypp" /* yacc.c:1648  */
    {(yyval.arithExp) = new ArithExp_Div((yyvsp[-2].arithExp), (yyvsp[0].arithExp), POS((yylsp[-1])));}
#line 2741 "parser.cpp" /* yacc.c:1648  */
    break;

  case 97:
#line 359 "parser.ypp" /* yacc.c:1648  */
    {(yyval.arithExp) = new ArithExp_Subtr(new ArithExp_Integer(0, POS((yylsp[-1]))), 
					 (yyvsp[0].arithExp), POS((yylsp[0])));}
#line 2748 "parser.cpp" /* yacc.c:1648  */
    break;

  case 98:
#line 362 "parser.ypp" /* yacc.c:1648  */
    {(yyval.arithExp) = new ArithExp_Integer(atoi((yyvsp[0].string)), POS((yylsp[0])));}
#line 2754 "parser.cpp" /* yacc.c:1648  */
    break;

  case 99:
#line 364 "parser.ypp" /* yacc.c:1648  */
    {(yyval.arithExp) = new ArithExp_Const((yyvsp[0].name), POS((yylsp[0])));}
#line 2760 "parser.cpp" /* yacc.c:1648  */
    break;

  case 100:
#line 366 "parser.ypp" /* yacc.c:1648  */
    {(yyval.arithExp) = (yyvsp[-1].arithExp);}
#line 2766 "parser.cpp" /* yacc.c:1648  */
    break;

  case 101:
#line 370 "parser.ypp" /* yacc.c:1648  */
    {(yyvsp[0].parDeclList)->push_front(new ParDecl(pPar0, (yyvsp[-2].name), NULL, POS((yylsp[-2])))); 
		 (yyval.parDeclList) = (yyvsp[0].parDeclList);}
#line 2773 "parser.cpp" /* yacc.c:1648  */
    break;

  case 102:
#line 373 "parser.ypp" /* yacc.c:1648  */
    {(yyvsp[0].parDeclList)->push_front(new ParDecl(pPar1, (yyvsp[-3].name), (yyvsp[-2].untypedExp), POS((yylsp[-3])))); 
		 (yyval.parDeclList) = (yyvsp[0].parDeclList);}
#line 2780 "parser.cpp" /* yacc.c:1648  */
    break;

  case 103:
#line 376 "parser.ypp" /* yacc.c:1648  */
    {(yyvsp[0].parDeclList)->push_front(new ParDecl(pPar2, (yyvsp[-3].name), (yyvsp[-2].untypedExp), POS((yylsp[-3])))); 
		 (yyval.parDeclList) = (yyvsp[0].parDeclList);}
#line 2787 "parser.cpp" /* yacc.c:1648  */
    break;

  case 104:
#line 379 "parser.ypp" /* yacc.c:1648  */
    {(yyvsp[0].parDeclList)->push_front(new ParDecl(pParU, (yyvsp[-2].name), NULL, POS((yylsp[-2])))); 
		 (yyval.parDeclList) = (yyvsp[0].parDeclList);}
#line 2794 "parser.cpp" /* yacc.c:1648  */
    break;

  case 105:
#line 382 "parser.ypp" /* yacc.c:1648  */
    {(yyvsp[0].parDeclList)->push_front(new ParDecl(pPar, (yyvsp[-3].name), (yyvsp[-2].untypedExp), POS((yylsp[-3])))); 
		 (yyval.parDeclList) = (yyvsp[0].parDeclList);}
#line 2801 "parser.cpp" /* yacc.c:1648  */
    break;

  case 106:
#line 385 "parser.ypp" /* yacc.c:1648  */
    {(yyval.parDeclList) = new ParDeclList(); 
		 (yyval.parDeclList)->push_front(new ParDecl(pPar0, (yyvsp[0].name), NULL, POS((yylsp[0]))));}
#line 2808 "parser.cpp" /* yacc.c:1648  */
    break;

  case 107:
#line 388 "parser.ypp" /* yacc.c:1648  */
    {(yyval.parDeclList) = new ParDeclList(); 
		 (yyval.parDeclList)->push_front(new ParDecl(pPar1, (yyvsp[-1].name), (yyvsp[0].untypedExp), POS((yylsp[-1]))));}
#line 2815 "parser.cpp" /* yacc.c:1648  */
    break;

  case 108:
#line 391 "parser.ypp" /* yacc.c:1648  */
    {(yyval.parDeclList) = new ParDeclList(); 
		 (yyval.parDeclList)->push_front(new ParDecl(pPar2, (yyvsp[-1].name), (yyvsp[0].untypedExp), POS((yylsp[-1]))));}
#line 2822 "parser.cpp" /* yacc.c:1648  */
    break;

  case 109:
#line 394 "parser.ypp" /* yacc.c:1648  */
    {(yyval.parDeclList) = new ParDeclList(); 
		 (yyval.parDeclList)->push_front(new ParDecl(pParU, (yyvsp[0].name), NULL, POS((yylsp[0]))));}
#line 2829 "parser.cpp" /* yacc.c:1648  */
    break;

  case 110:
#line 397 "parser.ypp" /* yacc.c:1648  */
    {(yyval.parDeclList) = new ParDeclList(); 
		 (yyval.parDeclList)->push_front(new ParDecl(pPar, (yyvsp[-1].name), (yyvsp[0].untypedExp), POS((yylsp[-1]))));}
#line 2836 "parser.cpp" /* yacc.c:1648  */
    break;

  case 111:
#line 402 "parser.ypp" /* yacc.c:1648  */
    {(yyvsp[0].bindExpList)->push_front((yyvsp[-2].bindExp)); 
		 (yyval.bindExpList) = (yyvsp[0].bindExpList);}
#line 2843 "parser.cpp" /* yacc.c:1648  */
    break;

  case 112:
#line 405 "parser.ypp" /* yacc.c:1648  */
    {(yyval.bindExpList) = new BindExpList(); 
		 (yyval.bindExpList)->push_front((yyvsp[0].bindExp));}
#line 2850 "parser.cpp" /* yacc.c:1648  */
    break;

  case 113:
#line 410 "parser.ypp" /* yacc.c:1648  */
    {(yyval.bindExp) = new BindExp((yyvsp[-2].name), (yyvsp[0].untypedExp), POS((yylsp[-1])));}
#line 2856 "parser.cpp" /* yacc.c:1648  */
    break;

  case 114:
#line 414 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExpList) = (yyvsp[0].untypedExpList);}
#line 2862 "parser.cpp" /* yacc.c:1648  */
    break;

  case 115:
#line 416 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExpList) = new UntypedExpList();}
#line 2868 "parser.cpp" /* yacc.c:1648  */
    break;

  case 116:
#line 420 "parser.ypp" /* yacc.c:1648  */
    {(yyvsp[0].untypedExpList)->push_front((yyvsp[-2].untypedExp));
                 (yyval.untypedExpList) = (yyvsp[0].untypedExpList);}
#line 2875 "parser.cpp" /* yacc.c:1648  */
    break;

  case 117:
#line 423 "parser.ypp" /* yacc.c:1648  */
    {(yyvsp[0].untypedExpList)->push_front(new UntypedExp_Interval(POS((yylsp[-2]))));
                 (yyval.untypedExpList) = (yyvsp[0].untypedExpList);}
#line 2882 "parser.cpp" /* yacc.c:1648  */
    break;

  case 118:
#line 426 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExpList) = new UntypedExpList();
                 (yyval.untypedExpList)->push_front((yyvsp[0].untypedExp));}
#line 2889 "parser.cpp" /* yacc.c:1648  */
    break;

  case 119:
#line 429 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExpList) = new UntypedExpList();
                 (yyval.untypedExpList)->push_front(new UntypedExp_Interval(POS((yylsp[0]))));}
#line 2896 "parser.cpp" /* yacc.c:1648  */
    break;

  case 120:
#line 434 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExpList) = (yyvsp[0].untypedExpList);}
#line 2902 "parser.cpp" /* yacc.c:1648  */
    break;

  case 121:
#line 436 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExpList) = NULL;}
#line 2908 "parser.cpp" /* yacc.c:1648  */
    break;

  case 122:
#line 440 "parser.ypp" /* yacc.c:1648  */
    {(yyvsp[0].untypedExpList)->push_front((yyvsp[-2].untypedExp)); 
		 (yyval.untypedExpList) = (yyvsp[0].untypedExpList);}
#line 2915 "parser.cpp" /* yacc.c:1648  */
    break;

  case 123:
#line 443 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExpList) = new UntypedExpList; 
		 (yyval.untypedExpList)->push_front((yyvsp[0].untypedExp));}
#line 2922 "parser.cpp" /* yacc.c:1648  */
    break;

  case 124:
#line 448 "parser.ypp" /* yacc.c:1648  */
    {(yyval.nameList) = (yyvsp[-1].nameList);}
#line 2928 "parser.cpp" /* yacc.c:1648  */
    break;

  case 125:
#line 450 "parser.ypp" /* yacc.c:1648  */
    {(yyval.nameList) = NULL;}
#line 2934 "parser.cpp" /* yacc.c:1648  */
    break;

  case 126:
#line 454 "parser.ypp" /* yacc.c:1648  */
    {(yyval.name) = new Name((yyvsp[0].string), POS((yylsp[0])));}
#line 2940 "parser.cpp" /* yacc.c:1648  */
    break;

  case 127:
#line 458 "parser.ypp" /* yacc.c:1648  */
    {(yyvsp[0].nameList)->push_front((yyvsp[-2].name)); 
		 (yyval.nameList) = (yyvsp[0].nameList);}
#line 2947 "parser.cpp" /* yacc.c:1648  */
    break;

  case 128:
#line 461 "parser.ypp" /* yacc.c:1648  */
    {(yyval.nameList) = new NameList(); 
		 (yyval.nameList)->push_front((yyvsp[0].name));}
#line 2954 "parser.cpp" /* yacc.c:1648  */
    break;

  case 129:
#line 466 "parser.ypp" /* yacc.c:1648  */
    {(yyvsp[0].varDeclList)->push_front(new VarDecl((yyvsp[-3].name), (yyvsp[-2].untypedExp), POS((yylsp[-3])))); 
		 (yyval.varDeclList) = (yyvsp[0].varDeclList);}
#line 2961 "parser.cpp" /* yacc.c:1648  */
    break;

  case 130:
#line 469 "parser.ypp" /* yacc.c:1648  */
    {(yyval.varDeclList) = new VarDeclList(); 
		 (yyval.varDeclList)->push_front(new VarDecl((yyvsp[-1].name), (yyvsp[0].untypedExp), POS((yylsp[-1]))));}
#line 2968 "parser.cpp" /* yacc.c:1648  */
    break;

  case 131:
#line 474 "parser.ypp" /* yacc.c:1648  */
    {(yyvsp[0].guideFuncList)->push_front((yyvsp[-2].guideFunc));
		 (yyval.guideFuncList) = (yyvsp[0].guideFuncList);}
#line 2975 "parser.cpp" /* yacc.c:1648  */
    break;

  case 132:
#line 477 "parser.ypp" /* yacc.c:1648  */
    {(yyval.guideFuncList) = new GuideFuncList();
		 (yyval.guideFuncList)->push_front((yyvsp[0].guideFunc));}
#line 2982 "parser.cpp" /* yacc.c:1648  */
    break;

  case 133:
#line 482 "parser.ypp" /* yacc.c:1648  */
    {(yyval.guideFunc) = new GuideFunc((yyvsp[-6].name), (yyvsp[-3].name), (yyvsp[-1].name));}
#line 2988 "parser.cpp" /* yacc.c:1648  */
    break;

  case 134:
#line 486 "parser.ypp" /* yacc.c:1648  */
    {(yyvsp[0].univList)->push_front((yyvsp[-2].univ)); 
		 (yyval.univList) = (yyvsp[0].univList);}
#line 2995 "parser.cpp" /* yacc.c:1648  */
    break;

  case 135:
#line 489 "parser.ypp" /* yacc.c:1648  */
    {(yyval.univList) = new UnivList(); 
		 (yyval.univList)->push_front((yyvsp[0].univ));}
#line 3002 "parser.cpp" /* yacc.c:1648  */
    break;

  case 136:
#line 494 "parser.ypp" /* yacc.c:1648  */
    {check_bits((yyvsp[0].string));
		 (yyval.univ) = new Univ((yyvsp[-2].name), (yyvsp[0].string), strlen((yyvsp[0].string)));}
#line 3009 "parser.cpp" /* yacc.c:1648  */
    break;

  case 137:
#line 497 "parser.ypp" /* yacc.c:1648  */
    {(yyval.univ) = new Univ((yyvsp[-2].name), (yyvsp[0].name));}
#line 3015 "parser.cpp" /* yacc.c:1648  */
    break;

  case 138:
#line 499 "parser.ypp" /* yacc.c:1648  */
    {(yyval.univ) = new Univ((yyvsp[0].name));}
#line 3021 "parser.cpp" /* yacc.c:1648  */
    break;

  case 139:
#line 503 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = (yyvsp[0].untypedExp);}
#line 3027 "parser.cpp" /* yacc.c:1648  */
    break;

  case 140:
#line 505 "parser.ypp" /* yacc.c:1648  */
    {(yyval.untypedExp) = NULL;}
#line 3033 "parser.cpp" /* yacc.c:1648  */
    break;

  case 141:
#line 509 "parser.ypp" /* yacc.c:1648  */
    {(yyvsp[0].mapList)->push_front(new ImportMap((yyvsp[-3].name), (yyvsp[-1].name), POS((yylsp[-2]))));
		 (yyval.mapList) = (yyvsp[0].mapList);}
#line 3040 "parser.cpp" /* yacc.c:1648  */
    break;

  case 142:
#line 512 "parser.ypp" /* yacc.c:1648  */
    {(yyval.mapList) = new ImportMapList();}
#line 3046 "parser.cpp" /* yacc.c:1648  */
    break;

  case 143:
#line 516 "parser.ypp" /* yacc.c:1648  */
    {(yyvsp[0].variantList)->push_front(new Variant((yyvsp[-5].name), (yyvsp[-3].componentList), POS((yylsp[-5])))); 
		 (yyval.variantList) = (yyvsp[0].variantList);}
#line 3053 "parser.cpp" /* yacc.c:1648  */
    break;

  case 144:
#line 519 "parser.ypp" /* yacc.c:1648  */
    {(yyvsp[0].variantList)->push_front(new Variant((yyvsp[-4].name), NULL, POS((yylsp[-4])))); 
		 (yyval.variantList) = (yyvsp[0].variantList);}
#line 3060 "parser.cpp" /* yacc.c:1648  */
    break;

  case 145:
#line 522 "parser.ypp" /* yacc.c:1648  */
    {(yyvsp[0].variantList)->push_front(new Variant((yyvsp[-2].name), NULL, POS((yylsp[-2])))); 
		 (yyval.variantList) = (yyvsp[0].variantList);}
#line 3067 "parser.cpp" /* yacc.c:1648  */
    break;

  case 146:
#line 525 "parser.ypp" /* yacc.c:1648  */
    {(yyval.variantList) = new VariantList();
		 (yyval.variantList)->push_front(new Variant((yyvsp[-3].name), (yyvsp[-1].componentList), POS((yylsp[-3]))));}
#line 3074 "parser.cpp" /* yacc.c:1648  */
    break;

  case 147:
#line 528 "parser.ypp" /* yacc.c:1648  */
    {(yyval.variantList) = new VariantList();
		 (yyval.variantList)->push_front(new Variant((yyvsp[-2].name), NULL, POS((yylsp[-2]))));}
#line 3081 "parser.cpp" /* yacc.c:1648  */
    break;

  case 148:
#line 531 "parser.ypp" /* yacc.c:1648  */
    {(yyval.variantList) = new VariantList();
		 (yyval.variantList)->push_front(new Variant((yyvsp[0].name), NULL, POS((yylsp[0]))));}
#line 3088 "parser.cpp" /* yacc.c:1648  */
    break;

  case 149:
#line 536 "parser.ypp" /* yacc.c:1648  */
    {(yyvsp[0].componentList)->push_front(new Component((yyvsp[-4].name), (yyvsp[-2].name), POS((yylsp[-4]))));
		 (yyval.componentList) = (yyvsp[0].componentList);}
#line 3095 "parser.cpp" /* yacc.c:1648  */
    break;

  case 150:
#line 539 "parser.ypp" /* yacc.c:1648  */
    {(yyval.componentList) = new ComponentList();
		 (yyval.componentList)->push_front(new Component((yyvsp[-2].name), (yyvsp[0].name), POS((yylsp[-2]))));}
#line 3102 "parser.cpp" /* yacc.c:1648  */
    break;

  case 151:
#line 544 "parser.ypp" /* yacc.c:1648  */
    {(yyval.constNode) = new ConstNode((yyvsp[-3].name), (yyvsp[-1].constNodeList), POS((yylsp[-3])));}
#line 3108 "parser.cpp" /* yacc.c:1648  */
    break;

  case 152:
#line 546 "parser.ypp" /* yacc.c:1648  */
    {(yyval.constNode) = new ConstNode((yyvsp[-2].name), NULL, POS((yylsp[-2])));}
#line 3114 "parser.cpp" /* yacc.c:1648  */
    break;

  case 153:
#line 548 "parser.ypp" /* yacc.c:1648  */
    {(yyval.constNode) = new ConstNode((yyvsp[0].name), NULL, POS((yylsp[0])));}
#line 3120 "parser.cpp" /* yacc.c:1648  */
    break;

  case 154:
#line 552 "parser.ypp" /* yacc.c:1648  */
    {(yyvsp[0].constNodeList)->push_front((yyvsp[-2].constNode)); (yyval.constNodeList) = (yyvsp[0].constNodeList);}
#line 3126 "parser.cpp" /* yacc.c:1648  */
    break;

  case 155:
#line 554 "parser.ypp" /* yacc.c:1648  */
    {(yyval.constNodeList) = new ConstNodeList;
		 (yyval.constNodeList)->push_front((yyvsp[0].constNode));}
#line 3133 "parser.cpp" /* yacc.c:1648  */
    break;

  case 156:
#line 559 "parser.ypp" /* yacc.c:1648  */
    {(yyval.string) = (yyvsp[0].string)+1;}
#line 3139 "parser.cpp" /* yacc.c:1648  */
    break;

  case 157:
#line 561 "parser.ypp" /* yacc.c:1648  */
    {(yyval.string) = 0;}
#line 3145 "parser.cpp" /* yacc.c:1648  */
    break;


#line 3149 "parser.cpp" /* yacc.c:1648  */
      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;
  *++yylsp = yyloc;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = (char *) YYSTACK_ALLOC (yymsg_alloc);
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }

  yyerror_range[1] = yylloc;

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval, &yylloc);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYTERROR;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;

      yyerror_range[1] = *yylsp;
      yydestruct ("Error: popping",
                  yystos[yystate], yyvsp, yylsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  yyerror_range[2] = yylloc;
  /* Using YYLLOC is tempting, but would change the location of
     the lookahead.  YYLOC is available though.  */
  YYLLOC_DEFAULT (yyloc, yyerror_range, 2);
  *++yylsp = yyloc;

  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined yyoverflow || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval, &yylloc);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  yystos[*yyssp], yyvsp, yylsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  return yyresult;
}
