	global _start

section .text

_start:
	endbr64
	push rbp
    mov rbp, rsp

    ;系统调用sys_write(stdout, hellomsg, strlen)
	mov rdi, 1          ;第一个参数，文件描述符，stdout是1
    mov rsi, hellomsg   ;第二个参数，字符串地址
    mov rdx, msglen     ;第三个参数，字符串长度
    mov rax, 1          ;系统调用号
    syscall

    leave

    ;系统调用sys_exit(0)
    xor rdi, rdi        ;参数，返回值0
    mov rax, 60         ;系统调用号
    syscall

	nop

section .data

hellomsg: db "Hello World!", 0xa
msglen: equ $-hellomsg
